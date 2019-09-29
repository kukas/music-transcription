import tensorflow as tf
import numpy as np

import datasets
from model import NetworkMelody, AdjustVoicingHook
from collections import namedtuple
import sys
import common

tf.logging.set_verbosity(tf.logging.ERROR)


def create_model(self, args):
    if args.spectrogram_undertone_stacking > 0 or args.spectrogram_overtone_stacking > 1:
        # pro spektrogramy začínající na nižší notě než je výstup
        # spectrogram_min_note = librosa.core.hz_to_midi(self.spectrogram_fmin)
        # offset = args.min_note - spectrogram_min_note
        spectrogram = common.harmonic_stacking(self, self.spectrogram, args.spectrogram_undertone_stacking, args.spectrogram_overtone_stacking)

    else:
        spectrogram = self.spectrogram[:, :, :self.bin_count, :]

    # if args.specaugment_prob:
        # in_shape = tf.shape(spectrogram)
        # batch_size = in_shape[0]
        # freq_shape = (batch_size, self.bin_count)
        # drop_batch = tf.random.uniform((batch_size, 1))
        # drop_freq_bands = tf.random.uniform((batch_size, 1), maxval=self.bin_count)

        # band_size = tf.random.uniform((batch_size, 1), minval=5, maxval=15)

        # masking_fn = tf.where(np.abs(tf.tile(tf.expand_dims(tf.range(self.bin_count, dtype=tf.float32), 0), [
        #                         batch_size, 1])-drop_freq_bands) < band_size, tf.zeros(freq_shape), tf.ones(freq_shape))

        # mask = tf.where(tf.tile(tf.greater(drop_batch, args.specaugment_prob), [1, self.bin_count]), tf.ones(freq_shape), masking_fn)
        # mask = tf.tile(mask[:, tf.newaxis, :, tf.newaxis], [1, in_shape[1], 1, in_shape[3]])

        # tf.summary.image("spectrogram", spectrogram[:,:,:,1:2])
        # tf.summary.image("spec_mask", mask[:,:,:,:1])
        # spectrogram = spectrogram*tf.cond(self.is_training, lambda: mask, lambda: tf.ones_like(spectrogram))
        # tf.summary.image("spectrogram_masked", spectrogram[:,:,:,:1])

    print("spectrogram shape", spectrogram.shape)

    args_context_size = int(self.context_width/self.spectrogram_hop_size)

    if args.activation is not None:
        activation = getattr(tf.nn, args.activation)

    with tf.name_scope('model_pitch'):
        layer = spectrogram

        if args.architecture.startswith("deep_hcnn"):
            assert len(args.conv_ctx) <= args.stacks
            # Prepare kernel sizes (time axis = audio context)
            args_ctx = np.abs(args.conv_ctx)
            args_dils = np.abs(args.dilations)
            ctxs = np.array([args_ctx[i] if i < len(args_ctx) else args_ctx[-1] for i in range(args.stacks)])
            dils = np.array([args_dils[i] if i < len(args_dils) else args_dils[-1] for i in range(args.stacks)])
            if args.conv_ctx[0] < 0:
                ctxs = np.array(list(reversed(ctxs)))
            if args.dilations[0] < 0:
                dils = np.array(list(reversed(dils)))
            print(ctxs)

            # Cut the unnecessary context
            needed_context_size = int(np.sum(np.ceil((ctxs-1)/2)) + np.ceil((args.last_conv_kernel[0]-1)/2))
            actual_context_size = args_context_size
            print("input context", args_context_size, "actual needed context", needed_context_size)
            if args_context_size < needed_context_size:
                print("Warning: provided context is shorter than the needed context field of the network")
            elif args_context_size > needed_context_size:
                if args.cut_context:
                    print("Cutting the unnecessary context {} --> ".format(layer.shape), end="")
                    diff = args_context_size - needed_context_size
                    layer = layer[:, diff:-diff, :, :]
                    actual_context_size -= diff
                    print(layer.shape, "context now:", actual_context_size)

            skip = None
            for i, conv_ctx, dil in zip(range(args.stacks), ctxs, dils):
                kernel = (conv_ctx, args.conv_range)
                print("add conv2d {} filters, {} kernel".format(args.filters, kernel))
                layer = tf.layers.conv2d(layer, args.filters, kernel, (1, 1), "same", activation=None, dilation_rate=(dil, 1))

                layer = activation(layer)

                if args.undertone_stacking > 0 or args.overtone_stacking > 1:
                    print("harmonic stacking {} --> ".format(layer.shape), end="")
                    layer = common.harmonic_stacking(self, layer, args.undertone_stacking, args.overtone_stacking)
                    print(layer.shape)

                layer = common.regularization(layer, args, training=self.is_training)

                if i < args.stacks - args.residual_end and i % args.residual_hop == 0:
                    if skip is None:
                        print(".- begin residual connection")
                    else:
                        print("|- adding residual connection")
                        layer += skip
                    skip = layer

            layer = tf.layers.conv2d(layer, 1, args.last_conv_kernel, (1, 1), "same", activation=None)
            if actual_context_size > 0:
                layer = layer[:, actual_context_size:-actual_context_size, :, :]

        self.note_logits = tf.squeeze(layer, -1)
        print("note_logits shape", self.note_logits.shape)

    if args.voicing:
        with tf.name_scope('model_voicing'):
            # Cut the unnecessary context
            voicing_layer = spectrogram
            if args_context_size > 0:
                voicing_layer = spectrogram[:, args_context_size:-args_context_size, :, :]

            if args.voicing_input == "only_salience":
                voicing_layer = tf.stop_gradient(layer)
            if args.voicing_input == "spectrogram_salience":
                voicing_layer = tf.concat([tf.stop_gradient(layer), voicing_layer], axis=-1)
            if args.voicing_input == "spectrogram_salience_train":
                voicing_layer = tf.concat([layer, voicing_layer], axis=-1)

            note = int(int(voicing_layer.shape[2])/6/12)

            voicing_layer = tf.layers.conv2d(voicing_layer, 64, (1, note), (1, 1), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            voicing_layer = tf.layers.conv2d(voicing_layer, 64, (1, note), (1, note), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            octave = int(int(voicing_layer.shape[2])/6)
            voicing_layer = tf.layers.conv2d(voicing_layer, 64, (1, octave), (1, 1), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            voicing_layer = tf.layers.conv2d(voicing_layer, 64, (1, octave), (1, octave), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            print("adding last conv valid layer")
            print("model output", voicing_layer.shape)
            voicing_layer = tf.layers.conv2d(voicing_layer, 1, (1, voicing_layer.shape[2]), (1, 1), "valid", activation=None, use_bias=True)
            print("last conv output", voicing_layer.shape)
            # print("cut context", voicing_layer.shape)
            self.voicing_logits = tf.squeeze(voicing_layer)
            print("squeeze", voicing_layer.shape)
    else:
        self.voicing_threshold = tf.Variable(0.15, trainable=False)
        tf.summary.scalar("model/voicing_threshold", self.voicing_threshold)


    self.loss = common.loss(self, args)
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

    hop_length = 512
    defaults = {
        # Change some of the common defaults
        "samplerate": 44100, "context_width": 10*hop_length, "annotations_per_window": 5, "hop_size": 1, "frame_width": hop_length,
        "note_range": 72, "min_note": 24, "evaluate_every": 5000, "evaluate_small_every": 1000, "annotation_smoothing": 0.18, "batch_size": 8,
        # Model specific defaults
        "learning_rate_decay_steps": 10000,
        "learning_rate_decay": 0.8,
        "spectrogram": "cqt",
        "spectrogram_top_db": 80,
        "spectrogram_filter_scale": 1.0,
        "spectrogram_undertone_stacking": 1,
        "spectrogram_overtone_stacking": 5,
        "cut_context": 1,
        "architecture": "deep_hcnn",
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
    common.name(args, specified_args, "spctrgrm")

    return args


def construct(args):
    network = NetworkMelody(args)

    with network.session.graph.as_default():
        spectrogram_function, spectrogram_thumb, spectrogram_info = common.spectrograms(args)
        # save spectrogram_thumb to hyperparams
        args.spectrogram_thumb = spectrogram_thumb

        def preload_fn(aa):
            aa.annotation = datasets.Annotation.from_time_series(*aa.annotation, hop_samples=args.frame_width*args.samplerate/44100)
            aa.audio.load_resampled_audio(args.samplerate).load_spectrogram(spectrogram_function, spectrogram_thumb, spectrogram_info[2])

        def dataset_transform(tf_dataset, dataset):
            return tf_dataset.map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size_evaluation).prefetch(10)

        def dataset_transform_train(tf_dataset, dataset):
            return tf_dataset.shuffle(10**5).map(dataset.prepare_example, num_parallel_calls=args.threads).map(dataset.specaugment(args), num_parallel_calls=args.threads).batch(args.batch_size).prefetch(10)

        train_dataset, test_datasets, validation_datasets = common.prepare_datasets(args.datasets, args, preload_fn, dataset_transform, dataset_transform_train)
        
        # Add voicing hook to the validation dataset
        if not args.voicing:
            for vd in validation_datasets:
                if not vd.name.startswith("small_"):
                    vd.hooks.append(AdjustVoicingHook())

        network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes, spectrogram_info=spectrogram_info)

    return network, train_dataset, validation_datasets, test_datasets


if __name__ == "__main__":
    common.main(sys.argv[1:], construct, parse_args)
