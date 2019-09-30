import tensorflow as tf
import numpy as np

import datasets
from model import NetworkMelody, AdjustVoicingHook
from collections import namedtuple
import sys
import common

import librosa

def harmonic_stacking(self, input, undertones, overtones, offset=0):
    spectrogram_windows = []
    print("stacking the spectrogram")
    for mult in [1/(x+2) for x in range(undertones)]+list(range(1, overtones+1)):
        f_ref = 440  # arbitrary reference frequency
        hz = f_ref*mult
        interval = librosa.core.hz_to_midi(hz) - librosa.core.hz_to_midi(f_ref)

        int_bins = int(round((interval + offset)*self.bins_per_semitone))

        start = max(int_bins, 0)
        end = self.bin_count+int_bins
        spec_layer = input[:, :, start:end, :]

        print(mult, "start", start, "end", end, "shape", spec_layer.shape, end=" ")

        if int_bins < 0:
            spec_layer = tf.pad(spec_layer, ((0, 0), (0, 0), (-int_bins, 0), (0, 0)))

        spec_layer = tf.pad(spec_layer, ((0, 0), (0, 0), (0, self.bin_count-spec_layer.shape[2]), (0, 0)))

        print("padded shape", spec_layer.shape)

        spectrogram_windows.append(spec_layer)
    return tf.concat(spectrogram_windows, axis=-1)

def create_model(self, args):
    spectrogram_min_note = librosa.core.hz_to_midi(self.spectrogram_fmin)
    if args.overtone_stacking > 0 or args.undertone_stacking > 0:
        # offset = args.min_note - spectrogram_min_note
        spectrogram = harmonic_stacking(self, self.spectrogram, args.undertone_stacking, args.overtone_stacking)

    else:
        spectrogram = self.spectrogram[:, :, :self.bin_count, :]

    # layer = tf.pad(layer, ((0, 0), (0, 0), (41, 41), (0, 0)))
    print(spectrogram.shape)

    context_size = int(self.context_width/self.spectrogram_hop_size)

    if args.activation is not None:
        activation = getattr(tf.nn, args.activation)

    with tf.name_scope('model_pitch'):
        layer = spectrogram

        if args.architecture == "bittner_improved":
            layer = tf.layers.conv2d(layer, 8*args.capacity_multiplier, (5, 5), (1, 1), "same", activation=None, use_bias=False)
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = activation(layer)
            layer = tf.layers.dropout(layer, args.dropout, training=self.is_training)
            residual = layer
            layer = tf.layers.conv2d(layer, 8*args.capacity_multiplier, (5, 5), (1, 1), "same", activation=None, use_bias=False)
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = activation(layer)
            layer = tf.layers.dropout(layer, args.dropout, training=self.is_training)
            residual += layer
            layer = tf.layers.conv2d(layer, 8*args.capacity_multiplier, (9, 3), (1, 1), "same", activation=None, use_bias=False)
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = activation(layer)
            layer = tf.layers.dropout(layer, args.dropout, training=self.is_training)
            residual += layer
            layer = tf.layers.conv2d(layer, 8*args.capacity_multiplier, (9, 3), (1, 1), "same", activation=None, use_bias=False)
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = activation(layer)
            layer = tf.layers.dropout(layer, args.dropout, training=self.is_training)
            residual += layer
            layer = tf.layers.conv2d(layer, 8*args.capacity_multiplier, (5, 70), (1, 1), "same", activation=None, use_bias=False)
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = activation(layer)
            layer = tf.layers.dropout(layer, args.dropout, training=self.is_training)
            residual += layer

            layer = residual

            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = tf.layers.conv2d(layer, 1, (10, 1), (1, 1), "same", activation=None, use_bias=False)
            layer_cut = layer[:, context_size:-context_size, :, :]

        if args.architecture == "bittnerlike":

            layer = tf.layers.conv2d(layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, 5), (1, 1), "same", activation=activation)
            layer = common.regularization(layer, args, training=self.is_training)
            residual = layer
            layer = tf.layers.conv2d(layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, 5), (1, 1), "same", activation=activation)
            layer = common.regularization(layer, args, training=self.is_training)
            residual += layer
            layer = tf.layers.conv2d(layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, 3), (1, 1), "same", activation=activation)
            layer = common.regularization(layer, args, training=self.is_training)
            residual += layer
            layer = tf.layers.conv2d(layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, 3), (1, 1), "same", activation=activation)
            layer = common.regularization(layer, args, training=self.is_training)
            residual += layer
            layer = tf.layers.conv2d(layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, 70), (1, 1), "same", activation=activation)
            layer = common.regularization(layer, args, training=self.is_training)
            residual += layer

            layer = residual

            layer = tf.layers.conv2d(layer, 1, (args.last_conv_ctx*2+1, 1), (1, 1), "same", activation=None)
            layer_cut = layer[:, context_size:-context_size, :, :]
        


        if args.architecture.startswith("deep_simple"):
            residual = None
            for i in range(args.stacks):
                layer = tf.layers.conv2d(layer, 8*args.capacity_multiplier, (args.conv_ctx, args.conv_range), (1, 1), "same", activation=None)

                layer = activation(layer)

                if args.harmonic_stacking:
                    layer = harmonic_stacking(self, layer, args.harmonic_stacking, args.harmonic_stacking+1)

                layer = common.regularization(layer, args, training=self.is_training)

                if residual is None:
                    residual = layer
                else:
                    residual += layer

            layer = residual

            layer = tf.layers.conv2d(layer, 1, (args.last_conv_ctx+1, 1), (1, 1), "same", activation=None)
            layer_cut = layer[:, context_size:-context_size, :, :]

        if args.architecture.startswith("deep_smooth"):
            residual = None
            ctx_end = 1
            dilations_start = 5
            for i in range(args.stacks):
                conv_ctx = args.conv_ctx if i < ctx_end or i >= dilations_start else 1
                dil_rate = (1, 1) if i < dilations_start else (2**(i-dilations_start), 1)
                layer = tf.layers.conv2d(layer, 8*args.capacity_multiplier, (conv_ctx, args.conv_range), (1, 1), "same", activation=None, dilation_rate=dil_rate)
                print(i, "kernel", (conv_ctx, args.conv_range), "dilation", dil_rate)

                layer = activation(layer)

                if args.harmonic_stacking:
                    layer = harmonic_stacking(self, layer, args.harmonic_stacking, args.harmonic_stacking+1)

                layer = common.regularization(layer, args, training=self.is_training)

                if residual is None:
                    residual = layer
                else:
                    residual += layer

            layer = residual

            layer = tf.layers.conv2d(layer, 1, (args.last_conv_ctx, 1), (1, 1), "same", activation=None)
            layer_cut = layer[:, context_size:-context_size, :, :]



        self.note_logits = tf.squeeze(layer_cut, -1)
        print("note_logits shape", self.note_logits.shape)

    if args.voicing:
        with tf.name_scope('model_voicing'):
            voicing_layer = tf.concat([tf.stop_gradient(layer), spectrogram], axis=-1)

            note = int(int(voicing_layer.shape[2])/6/12)

            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.voicing_capacity_multiplier, (args.voicing_conv_ctx*2+1, note), (1, 1), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.voicing_capacity_multiplier, (args.voicing_conv_ctx*2+1, note), (1, note), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            octave = int(int(voicing_layer.shape[2])/6)
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.voicing_capacity_multiplier, (args.voicing_conv_ctx*2+1, octave), (1, 1), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.voicing_capacity_multiplier, (args.voicing_conv_ctx*2+1, octave), (1, octave), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            print("adding last conv valid layer")
            print("model output", voicing_layer.shape)
            if args.voicing_last_conv_ctx:
                voicing_layer = tf.pad(voicing_layer, ((0, 0), (args.voicing_last_conv_ctx, args.voicing_last_conv_ctx), (0, 0), (0, 0)))
                print("padded", voicing_layer.shape)
            voicing_layer = tf.layers.conv2d(voicing_layer, 1, (args.voicing_last_conv_ctx*2+1, voicing_layer.shape[2]), (1, 1), "valid", activation=None, use_bias=True)
            print("last conv output", voicing_layer.shape)
            voicing_layer = voicing_layer[:, context_size:-context_size, :, :]
            print("cut context", voicing_layer.shape)
            self.voicing_logits = tf.squeeze(voicing_layer)
            print("squeeze", voicing_layer.shape)
    else:
        self.voicing_threshold = tf.Variable(0.15, trainable=False)
        tf.summary.scalar("model/voicing_threshold", self.voicing_threshold)

    self.loss = common.loss(self, args)
    self.est_notes = common.est_notes(self, args)
    self.training = common.optimizer(self, args)


def parse_args(argv):
    hop_length = 512
    parser = common.common_arguments({
        "samplerate": 44100, "context_width": 10*hop_length, "annotations_per_window": 20, "hop_size": 1, "frame_width": hop_length,
        "note_range": 72, "min_note": 24,
        "evaluate_every": 5000,
        "evaluate_small_every": 1000,
        "annotation_smoothing": 0.18,
        "learning_rate_decay_steps": 10000,
        "learning_rate_decay": 0.8
    })
    # Model specific arguments
    parser.add_argument("--spectrogram", default="cqt_fs", type=str, help="Postprocessing layer")
    parser.add_argument("--architecture", default="bittnerlike", type=str, help="Postprocessing layer")
    parser.add_argument("--capacity_multiplier", default=8, type=int, help="Capacity")
    
    parser.add_argument("--stacks", default=10, type=int, help="Stacks")
    parser.add_argument("--conv_range", default=3, type=int, help="Stack kernel width")
    parser.add_argument("--harmonic_stacking", default=1, type=int, help="harmonic stacking undertones and overtones")


    parser.add_argument("--voicing_capacity_multiplier", default=8, type=int, help="Capacity")
    parser.add_argument("--undertone_stacking", default=5, type=int, help="spectrogram stacking")
    parser.add_argument("--overtone_stacking", default=10, type=int, help="spectrogram stacking")

    parser.add_argument("--voicing", action='store_true', help="Add voicing model.")

    parser.add_argument("--conv_ctx", default=0, type=int)
    parser.add_argument("--last_conv_ctx", default=0, type=int)
    parser.add_argument("--voicing_conv_ctx", default=0, type=int)
    parser.add_argument("--voicing_last_conv_ctx", default=0, type=int)
    parser.add_argument("--batchnorm", default=0, type=int)
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--activation", default="relu", type=str)


    args = parser.parse_args(argv)

    common.name(args, "cqt_voicing_residual_batchnorm")

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
