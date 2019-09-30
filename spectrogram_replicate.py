import tensorflow as tf
import numpy as np

import datasets
from model import NetworkMelody, AdjustVoicingHook
from collections import namedtuple
import sys
import common

import librosa


def create_model(self, args):
    layer = self.spectrogram

    print(layer.shape)

    context_size = int(args.context_width/self.spectrogram_hop_size)

    with tf.name_scope('model_pitch'):
        layer = tf.layers.conv2d(layer, 8*args.capacity_multiplier, (5, 5), (1, 1), "same", activation=None, use_bias=False)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.nn.relu(layer)
        layer = tf.layers.dropout(layer, 0.25, training=self.is_training)
        residual = layer
        layer = tf.layers.conv2d(layer, 8*args.capacity_multiplier, (5, 5), (1, 1), "same", activation=None, use_bias=False)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.nn.relu(layer)
        layer = tf.layers.dropout(layer, 0.25, training=self.is_training)
        residual += layer
        layer = tf.layers.conv2d(layer, 8*args.capacity_multiplier, (9, 3), (1, 1), "same", activation=None, use_bias=False)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.nn.relu(layer)
        layer = tf.layers.dropout(layer, 0.25, training=self.is_training)
        residual += layer
        layer = tf.layers.conv2d(layer, 8*args.capacity_multiplier, (9, 3), (1, 1), "same", activation=None, use_bias=False)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.nn.relu(layer)
        layer = tf.layers.dropout(layer, 0.25, training=self.is_training)
        residual += layer
        layer = tf.layers.conv2d(layer, 8*args.capacity_multiplier, (5, 70), (1, 1), "same", activation=None, use_bias=False)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.nn.relu(layer)
        layer = tf.layers.dropout(layer, 0.25, training=self.is_training)
        residual += layer

        layer = residual

        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.layers.conv2d(layer, 1, (10, 1), (1, 1), "same", activation=None, use_bias=False)
        layer_cut = layer[:, context_size:-context_size, :, :]
        # layer = tf.layers.conv2d(layer, 1, (10, 1), (1, 1), "same", activation=None, use_bias=True)

        note_output = tf.squeeze(layer_cut, -1)
        print(note_output.shape)
        self.note_logits = note_output

    if args.voicing:
        with tf.name_scope('model_voicing'):
            voicing_input = tf.concat([tf.stop_gradient(layer), self.spectrogram], axis=-1)
            # voicing_input = spectrogram
            print(voicing_input.shape)
            voicing_layer = tf.layers.conv2d(voicing_input, 64, (5, 5), (1, 1), "same", activation=tf.nn.relu, use_bias=False)
            voicing_layer = tf.layers.dropout(voicing_layer, 0.25, training=self.is_training)
            voicing_layer = tf.layers.batch_normalization(voicing_layer, training=self.is_training)

            voicing_layer = tf.layers.conv2d(voicing_layer, 64, (5, 70), (1, 5), "same", activation=tf.nn.relu, use_bias=False)
            voicing_layer = tf.layers.dropout(voicing_layer, 0.25, training=self.is_training)
            voicing_layer = tf.layers.batch_normalization(voicing_layer, training=self.is_training)

            voicing_layer = tf.layers.conv2d(voicing_layer, 64, (5, 12), (1, 12), "same", activation=tf.nn.relu, use_bias=False)
            voicing_layer = tf.layers.dropout(voicing_layer, 0.25, training=self.is_training)
            voicing_layer = tf.layers.batch_normalization(voicing_layer, training=self.is_training)

            voicing_layer = tf.layers.conv2d(voicing_layer, 64, (15, 3), (1, 1), "same", activation=tf.nn.relu, use_bias=False)
            voicing_layer = tf.layers.dropout(voicing_layer, 0.25, training=self.is_training)
            voicing_layer = tf.layers.batch_normalization(voicing_layer, training=self.is_training)

            print(voicing_layer.shape)
            voicing_layer = tf.layers.conv2d(voicing_layer, 1, (1, 6), (1, 1), "valid", activation=None, use_bias=True)
            cut_layer = voicing_layer[:, context_size:-context_size, :, :]
            print(cut_layer.shape)
            self.voicing_logits = tf.squeeze(cut_layer)
    else:
        self.voicing_threshold = tf.Variable(0.15, trainable=False)
        tf.summary.scalar("model/voicing_threshold", self.voicing_threshold)

    self.loss = common.loss(self, args)
    self.est_notes = common.est_notes(self, args)
    self.training = common.optimizer(self, args)



def parse_args(argv):
    hop_length = 512
    parser = common.common_arguments({
        "samplerate": 44100, "context_width": 14*hop_length, "annotations_per_window": 20, "hop_size": 1, "frame_width": hop_length,
        "note_range": 72, "min_note": 24,
        "batch_size": 8,
        "evaluate_every": 5000,
        "evaluate_small_every": 1000,
        "annotation_smoothing": 0.177,
    })
    # Model specific arguments
    parser.add_argument("--spectrogram", default="hcqt", type=str, help="Postprocessing layer")
    parser.add_argument("--capacity_multiplier", default=8, type=int, help="Capacity")
    parser.add_argument("--voicing", action='store_true', help="Add voicing model.")

    args = parser.parse_args(argv)

    common.name(args, "cqt_voicing_residual_batchnorm")

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

        if not args.voicing:
            for vd in validation_datasets:
                if not vd.name.startswith("small_"):
                    vd.hooks.append(AdjustVoicingHook())

        network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes, spectrogram_info=spectrogram_info)

    return network, train_dataset, validation_datasets, test_datasets


if __name__ == "__main__":
    common.main(sys.argv[1:], construct, parse_args)
