import tensorflow as tf
import numpy as np

import datasets
from model import NetworkMelody, AdjustVoicingHook
from collections import namedtuple
import sys
import common

import librosa

from tensorflow.python.ops import array_ops

def create_model(self, args):
    if args.spectrogram_undertone_stacking > 0 or args.spectrogram_overtone_stacking > 1:
        # for spectrograms where the min. frequency doesn't correspond to output min. note
        # spectrogram_min_note = librosa.core.hz_to_midi(self.spectrogram_fmin)
        # offset = args.min_note - spectrogram_min_note
        spectrogram = common.harmonic_stacking(self, self.spectrogram, args.spectrogram_undertone_stacking, args.spectrogram_overtone_stacking)

    else:
        spectrogram = self.spectrogram[:, :, :self.bin_count, :]


    args_context_size = int(self.context_width/self.spectrogram_hop_size)

    if args.activation is not None:
        activation = getattr(tf.nn, args.activation)

    with tf.name_scope('model_pitch'):
        layer = spectrogram

        if args.architecture.startswith("deep_simple"):
            residual = None
            for i in range(args.stacks):
                layer = tf.layers.conv2d(layer, args.filters, (args.conv_ctx[0], args.conv_range), (1, 1), "same", activation=None)

                layer = activation(layer)

                if args.undertone_stacking > 0 or args.overtone_stacking > 1:
                    print("harmonic stacking {} --> ".format(layer.shape), end="")
                    layer = common.harmonic_stacking(self, layer, args.undertone_stacking, args.overtone_stacking)
                    print(layer.shape)

                layer = common.regularization(layer, args, training=self.is_training)

                if residual is None:
                    residual = layer
                else:
                    residual += layer

            layer = residual

            layer = tf.layers.conv2d(layer, 1, args.last_conv_kernel, (1, 1), "same", activation=None)
            layer_cut = layer[:, args_context_size:-args_context_size, :, :]
            self.note_logits = tf.squeeze(layer_cut, -1)
            print("note_logits shape", self.note_logits.shape)

    if args.voicing:
        raise NotImplementedError
    else:
        self.voicing_threshold = tf.Variable(0.15, trainable=False)
        tf.summary.scalar("model/voicing_threshold", self.voicing_threshold)

    # multif0 loss ---------
    with tf.name_scope("losses"):

        annotations = self.annotations - args.min_note

        voicing_ref = tf.cast(tf.greater(annotations[:, :, 0], 0), tf.float32)
        loss_names = []
        losses = []
        if self.note_logits is not None:
            if args.annotation_smoothing > 0:
                self.note_probabilities = tf.nn.sigmoid(self.note_logits)
                annotations_per_frame = tf.shape(annotations)[-1]
                note_bins = tf.tile(tf.expand_dims(self.note_bins, 2), [1, 1, annotations_per_frame, 1])
                note_ref = tf.tile(tf.reshape(annotations, [-1, self.annotations_per_window, annotations_per_frame, 1]), [1, 1, 1, self.bin_count])
                
                ref_probabilities = tf.exp(-(note_ref-note_bins)**2/(2*args.annotation_smoothing**2))
                # ref_probabilities = tf.concat([ref_probabilities[:, :, :1, :], ref_probabilities[:, :, 1:, :]*args.miss_weight], axis=2)
                ref_probabilities = tf.reduce_sum(ref_probabilities, axis=2)

                # self.note_probabilities = ref_probabilities
                # print(ref_probabilities.eval(), ref_probabilities.shape)

                unvoiced_weights = (1-voicing_ref)*args.unvoiced_loss_weight
                voicing_weights = tf.tile(tf.expand_dims(voicing_ref+unvoiced_weights, -1), [1, 1, self.bin_count])

                note_loss = tf.losses.sigmoid_cross_entropy(ref_probabilities, self.note_logits, weights=voicing_weights)


            
            loss_names.append("note_loss")
            losses.append(note_loss)

        if args.l2_loss_weight > 0:
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.reduce_sum(tf.constant(args.l2_loss_weight)*reg_variables)

            loss_names.append("l2_loss")
            losses.append(l2_loss)

        if self.voicing_logits is not None:
            voicing_loss = tf.losses.sigmoid_cross_entropy(voicing_ref, self.voicing_logits)

            loss_names.append("voicing_loss")
            losses.append(voicing_loss)

    if len(losses) > 1:
        for name, loss in zip(loss_names, losses):
            tf.summary.scalar('metrics/train/'+name, loss)

    self.loss = tf.math.add_n(losses)
    self.est_notes = common.est_notes(self, args)
    self.training = common.optimizer(self, args)



def parse_args(argv):
    parser = common.common_arguments_parser()
    # Model specific arguments
    # input
    parser.add_argument("--spectrogram", type=str, help="Spectrogram method")
    parser.add_argument("--spectrogram_top_db", type=float, help="Spectrogram top_db")
    parser.add_argument("--spectrogram_filter_scale", type=float, help="Spectrogram filter_scale")
    parser.add_argument("--spectrogram_undertone_stacking", type=int, help="spectrogram undertone stacking")
    parser.add_argument("--spectrogram_overtone_stacking", type=int, help="spectrogram overtone stacking")
    parser.add_argument("--cut_context", type=int, help="Cut unnecessary context, doesn't work with dilations!")
    # model
    parser.add_argument("--architecture", type=str, help="Model architecture")
    parser.add_argument("--filters", type=int, help="Filters in convolutions")
    parser.add_argument("--stacks", type=int, help="Stacks")
    parser.add_argument("--conv_range", type=int, help="Stack kernel size in frequency axis")
    parser.add_argument("--undertone_stacking", type=int, help="Undertone stacking in the model")
    parser.add_argument("--overtone_stacking", type=int, help="Overtone stacking in the model")
    parser.add_argument("--activation", type=str, help="Activation function for the convolution stack")
    # context
    parser.add_argument("--conv_ctx", nargs="+", type=int, help="Stack kernel sizes in time axis")
    parser.add_argument("--dilations", nargs="+", type=int, help="Dilation rate for the convolutions")
    parser.add_argument("--last_conv_kernel", nargs=2, type=int)
    # residual
    parser.add_argument("--residual_hop", type=int, help="Size of one block around which there is a residual connection")
    parser.add_argument("--residual_end", type=int, help="No residual connection in last N layers")
    parser.add_argument("--residual_op", type=str, help="Residual connection operation (add for ResNet, concat for DenseNet)")
    # regularization
    parser.add_argument("--batchnorm", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--specaugment_prob", type=float)
    parser.add_argument("--specaugment_freq_mask_num", type=int)
    parser.add_argument("--specaugment_freq_mask_max", type=int)
    parser.add_argument("--specaugment_time_mask_num", type=int)
    parser.add_argument("--specaugment_time_mask_max", type=int)
    # voicing module
    parser.add_argument("--voicing", type=int)
    parser.add_argument("--voicing_input", type=str)

    args = parser.parse_args(argv)

    hop_length = 256
    defaults = {
        # Change some of the common defaults
        "samplerate": 44100, "context_width": 10*hop_length, "annotations_per_window": 5, "hop_size": 1, "frame_width": hop_length,
        "note_range": 72, "min_note": 24, "evaluate_every": 5000, "evaluate_small_every": 1000, "annotation_smoothing": 0.18, "batch_size": 8,
        "unvoiced_loss_weight": 1.0,
        "datasets": ["mdb_mel4"],
        # Model specific defaults
        "learning_rate_decay_steps": 10000,
        "learning_rate_decay": 0.8,
        "spectrogram": "cqt",
        "spectrogram_top_db": 80,
        "spectrogram_filter_scale": 1.0,
        "spectrogram_undertone_stacking": 1,
        "spectrogram_overtone_stacking": 5,
        "cut_context": 1,
        "architecture": "deep_simple",
        "filters": 16,
        "stacks": 10,
        "conv_range": 3,
        "undertone_stacking": 0,
        "overtone_stacking": 1,
        "activation": "relu",
        "conv_ctx": [1],
        "dilations": [1],
        "last_conv_kernel": [1, 1],
        "residual_hop": 1,
        "residual_end": 0,
        "residual_op": "add",
        "batchnorm": 0,
        "dropout": 0.3,
        "specaugment_prob": 0.0,
        "specaugment_freq_mask_num": 2,
        "specaugment_freq_mask_max": 27,
        "specaugment_time_mask_num": 1,
        "specaugment_time_mask_max": 5,
        "voicing": 0,
        "voicing_input": "spectrogram_salience",
    }
    specified_args = common.argument_defaults(args, defaults)
    common.name(args, specified_args, "mf0")

    return args

def construct(args):
    network = NetworkMelody(args)

    with network.session.graph.as_default():
        spectrogram_function, spectrogram_thumb, spectrogram_info = common.spectrograms(args)
        # save spectrogram_thumb to hyperparams
        args.spectrogram_thumb = spectrogram_thumb

        def preload_fn(aa):
            annot_path, uid = aa.annotation
            if uid.startswith("mdb_"):
                uid = uid + "_mel4"
            aa.annotation = datasets.Annotation.from_time_series(annot_path, uid, hop_samples=args.frame_width*args.samplerate/44100, unique_mf0=True)
            aa.audio.load_resampled_audio(args.samplerate).load_spectrogram(spectrogram_function, spectrogram_thumb, spectrogram_info[2])

        def dataset_transform(tf_dataset, dataset):
            return tf_dataset.map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size_evaluation).prefetch(10)

        def dataset_transform_train(tf_dataset, dataset):
            return tf_dataset.shuffle(10**5).map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size).prefetch(10)

        train_dataset, test_datasets, validation_datasets = common.prepare_datasets(args.datasets, args, preload_fn, dataset_transform, dataset_transform_train)
        
        if not args.voicing:
            for vd in validation_datasets:
                if not vd.name.startswith("small_"):
                    vd.hooks.append(AdjustVoicingHook())

        network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes, spectrogram_info=spectrogram_info)

    return network, train_dataset, validation_datasets, test_datasets


if __name__ == "__main__":
    common.main(sys.argv[1:], construct, parse_args)
