import tensorflow as tf
import numpy as np

import datasets
from model import NetworkMelody
from collections import namedtuple
import sys
import common

import librosa

def create_model(self, args):
    context_size = int(self.context_width/self.spectrogram_hop_size)

    with tf.name_scope('model_pitch'):
        self.note_logits = None
        self.note_probabilities = self.spectrogram[:, context_size:-context_size, :360, 0]
    

    with tf.name_scope('model_voicing'):
        # voicing_layer = tf.concat([tf.stop_gradient(layer), spectrogram], axis=-1)

        if args.harmonic_stacking > 1:
            spectrogram_windows = []
            print("stacking the spectrogram")
            for i in range(args.harmonic_stacking):
                f_ref = 440 # arbitrary reference frequency
                hz = f_ref*(i+1)
                interval = librosa.core.hz_to_midi(hz) - librosa.core.hz_to_midi(f_ref)
                int_bins = int(round(interval*self.bins_per_semitone))
                spec_layer = self.spectrogram[:, :, int_bins:self.bin_count+int_bins, :]
                print(i+1, "offset", int_bins, "end", self.bin_count+int_bins, "shape", spec_layer.shape)
                spec_layer = tf.pad(spec_layer, ((0, 0), (0, 0), (0, self.bin_count-spec_layer.shape[2]), (0, 0)))
                print("padded shape", spec_layer.shape)
                spectrogram_windows.append(spec_layer)
            voicing_layer = tf.concat(spectrogram_windows, axis=-1)

        else:
            voicing_layer = self.spectrogram[:, :, :360, :]

        if args.first_pool_type == "avg":
            voicing_layer = tf.layers.average_pooling2d(voicing_layer, args.first_pool_size, args.first_pool_stride, padding="same")
        if args.first_pool_type == "max":
            voicing_layer = tf.layers.max_pooling2d(voicing_layer, args.first_pool_size, args.first_pool_stride, padding="same")
        
        print("after pooling", voicing_layer.shape)

        octave = int(int(voicing_layer.shape[2])/6)
        note = int(int(voicing_layer.shape[2])/6/12)

        if args.activation is not None:
            activation = getattr(tf.nn, args.activation)

        if args.architecture == "full_1layer":
            if args.conv_ctx:
                voicing_layer = tf.pad(voicing_layer, ((0, 0), (args.conv_ctx, args.conv_ctx), (0, 0), (0, 0)))
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, voicing_layer.shape[2]), (1, 1), "valid", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)
        if args.architecture == "octave_1layer":
            if args.conv_ctx:
                voicing_layer = tf.pad(voicing_layer, ((0, 0), (args.conv_ctx, args.conv_ctx), (0, 0), (0, 0)))
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, octave), (1, octave), "valid", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)
        if args.architecture == "note_1layer":
            if args.conv_ctx:
                voicing_layer = tf.pad(voicing_layer, ((0, 0), (args.conv_ctx, args.conv_ctx), (0, 0), (0, 0)))
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, note), "valid", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)
        
        if args.architecture == "octave_octave":
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, octave), (1, octave), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, octave), (1, octave), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)
        
        if args.architecture == "note_note":
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, note), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, note), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)
        
        if args.architecture == "note_dilated":
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, note), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (1, 6), (1, 1), "same", activation=activation, dilation_rate=(1, octave))
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

        if args.architecture == "dilated_note":
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (1, 6), (1, 1), "same", activation=activation, dilation_rate=(1, octave))
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, note), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

        if args.architecture == "note_octave":
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, note), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, octave), (1, octave), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

        if args.architecture == "octave_note":
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, octave), (1, octave), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, note), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)
        

        if args.architecture == "note_octave_fix":
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, note), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)
            octave = int(int(voicing_layer.shape[2])/6)
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, octave), (1, octave), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

        if args.architecture == "note_note_octave":
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, 1), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, note), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            octave = int(int(voicing_layer.shape[2])/6)
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, octave), (1, octave), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

        if args.architecture == "note_note_octave_octave":
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, 1), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, note), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            octave = int(int(voicing_layer.shape[2])/6)
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, octave), (1, 1), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, octave), (1, octave), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

        if args.architecture == "note_note_note_octave":
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, 1), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, 1), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, note), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            octave = int(int(voicing_layer.shape[2])/6)

            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, octave), (1, octave), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

        if args.architecture == "note_note_note_octave_octave":
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, 1), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, 1), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, note), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            octave = int(int(voicing_layer.shape[2])/6)
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, octave), (1, 1), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, octave), (1, octave), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)
        
        if args.architecture == "note_octave_octave_temporal":
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, note), (1, 1), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            octave = int(int(voicing_layer.shape[2])/6)
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, octave), (1, note), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            octave = int(int(voicing_layer.shape[2])/6)
            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*2+1, octave), (1, octave), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)

            voicing_layer = tf.layers.conv2d(voicing_layer, 8*args.capacity_multiplier, (args.conv_ctx*7+1, 3), (1, 1), "same", activation=activation)
            voicing_layer = common.regularization(voicing_layer, args, training=self.is_training)



        if args.last_layer == "conv":
            print("adding last conv valid layer")
            print("model output", voicing_layer.shape)
            if args.last_conv_ctx:
                voicing_layer = tf.pad(voicing_layer, ((0, 0), (args.last_conv_ctx, args.last_conv_ctx), (0, 0), (0, 0)))
                print("padded", voicing_layer.shape)
            voicing_layer = tf.layers.conv2d(voicing_layer, 1, (args.last_conv_ctx*2+1, voicing_layer.shape[2]), (1, 1), "valid", activation=None, use_bias=True)
            print("last conv output", voicing_layer.shape)
            voicing_layer = voicing_layer[:, context_size:-context_size, :, :]
            print("cut context", voicing_layer.shape)
            self.voicing_logits = tf.squeeze(voicing_layer)
            print("squeeze", voicing_layer.shape)
        if args.last_layer == "dense":
            voicing_layer = tf.layers.flatten(voicing_layer)
            self.voicing_logits = tf.layers.dense(voicing_layer, args.annotations_per_window)

    self.loss = common.loss(self, args)
    self.est_notes = common.est_notes(self, args)
    self.training = common.optimizer(self, args)

HOP_LENGTH = 512

def parse_args(argv):
    parser = common.common_arguments({
        "samplerate": 44100, "context_width": 5*HOP_LENGTH, "annotations_per_window": 10, "hop_size": 1, "frame_width": HOP_LENGTH,
        "note_range": 72, "min_note": 24, "batch_size": 32,
        "evaluate_every": 5000,
        "evaluate_small_every": 1000,
    })
    # Model specific arguments
    parser.add_argument("--spectrogram", default="cqt", type=str, help="Postprocessing layer")

    parser.add_argument("--first_pool_type", default=None, type=str, help="First pooling type")
    parser.add_argument("--first_pool_size", default=[1, 5], nargs="+", type=str, help="Input pooling size")
    parser.add_argument("--first_pool_stride", default=[1, 5], nargs="+", type=str, help="Input pooling stride")
    parser.add_argument("--capacity_multiplier", default=8, type=int)
    parser.add_argument("--architecture", default="full_1layer", type=str)
    parser.add_argument("--conv_ctx", default=0, type=int)
    parser.add_argument("--batchnorm", default=0, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--last_layer", default="conv", type=str)
    parser.add_argument("--last_conv_ctx", default=0, type=int)
    parser.add_argument("--harmonic_stacking", default=1, type=int)
    parser.add_argument("--activation", default="relu", type=str)

    args = parser.parse_args(argv)

    common.name(args, "voicing")

    return args

def construct(args):
    network = NetworkMelody(args)

    with network.session.graph.as_default():
        spectrogram_function, spectrogram_thumb, spectrogram_info = common.spectrograms(args)

        def preload_fn(aa):
            aa.annotation = datasets.Annotation.from_time_series(*aa.annotation, args.frame_width*args.samplerate/44100)
            aa.audio.load_resampled_audio(args.samplerate).load_spectrogram(spectrogram_function, spectrogram_thumb, spectrogram_info[2])

        def dataset_transform(tf_dataset, dataset):
            return tf_dataset.map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size_evaluation).prefetch(10)

        def dataset_transform_train(tf_dataset, dataset):
            return tf_dataset.shuffle(10**5).map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size).prefetch(10)

        train_dataset, test_datasets, validation_datasets = common.prepare_datasets(args.datasets, args, preload_fn, dataset_transform, dataset_transform_train)

        network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes, spectrogram_info=spectrogram_info)

    return network, train_dataset, validation_datasets, test_datasets


if __name__ == "__main__":
    common.main(sys.argv[1:], construct, parse_args)
