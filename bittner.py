import tensorflow as tf

import numpy as np

import datasets
from model import NetworkMelody, AdjustVoicingHook
from collections import namedtuple
import sys
import common

import librosa
import evaluation

from model import VD, VisualOutputHook, MetricsHook, SaveBestModelHook, EvaluationHook, CSVOutputWriterHook

def create_model(self, args):
    spectrogram = self.spectrogram

    with tf.name_scope('model'):

        layer = spectrogram

        if args.spectrogram_undertone_stacking > 0 or args.spectrogram_overtone_stacking > 1:
            layer = common.harmonic_stacking(self, layer, args.spectrogram_undertone_stacking, args.spectrogram_overtone_stacking)

        layer = tf.layers.batch_normalization(layer, training=self.is_training)

        layer = tf.layers.conv2d(layer, 2*args.capacity_multiplier, (5, 5), (1, 1), "same", activation=tf.nn.relu)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)

        if args.undertone_stacking > 0 or args.overtone_stacking > 1:
            layer = common.harmonic_stacking(self, layer, args.undertone_stacking, args.overtone_stacking)

        layer = tf.layers.conv2d(layer, args.capacity_multiplier, (5, 5), (1, 1), "same", activation=tf.nn.relu)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)

        if args.undertone_stacking > 0 or args.overtone_stacking > 1:
            layer = common.harmonic_stacking(self, layer, args.undertone_stacking, args.overtone_stacking)

        layer = tf.layers.conv2d(layer, args.capacity_multiplier, (3, 3), (1, 1), "same", activation=tf.nn.relu)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)

        if args.undertone_stacking > 0 or args.overtone_stacking > 1:
            layer = common.harmonic_stacking(self, layer, args.undertone_stacking, args.overtone_stacking)

        layer = tf.layers.conv2d(layer, args.capacity_multiplier, (3, 3), (1, 1), "same", activation=tf.nn.relu)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)

        if args.undertone_stacking > 0 or args.overtone_stacking > 1:
            layer = common.harmonic_stacking(self, layer, args.undertone_stacking, args.overtone_stacking)

        layer = tf.layers.conv2d(layer, max(1, args.capacity_multiplier//8), (3, 70), (1, 1), "same", activation=tf.nn.relu)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)

        if args.undertone_stacking > 0 or args.overtone_stacking > 1:
            layer = common.harmonic_stacking(self, layer, args.undertone_stacking, args.overtone_stacking)

        layer = tf.layers.conv2d(layer, 1, (1, 1), (1, 1), "same", activation=tf.nn.sigmoid)

        note_output = tf.squeeze(layer, -1)
        self.note_logits = note_output
        self.note_probabilities = note_output

    annotations = self.annotations[:, :, 0] - args.min_note
    voicing_ref = tf.cast(tf.greater(annotations, 0), tf.float32)

    note_ref = tf.tile(tf.reshape(annotations, [-1, self.annotations_per_window, 1]), [1, 1, self.bin_count])
    ref_probabilities = tf.exp(-(note_ref-self.note_bins)**2/(args.annotation_smoothing**2))

    voicing_weights = tf.tile(tf.expand_dims(voicing_ref, -1), [1, 1, self.bin_count])
    ref_probabilities *= voicing_weights

    def bkld(y_true, y_pred):
        """KL Divergence where both y_true an y_pred are probabilities
        """
        epsilon = tf.constant(1e-07)
        y_true = tf.clip_by_value(y_true, epsilon, 1.0 - epsilon)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        return tf.math.reduce_mean(-1.0*y_true * tf.log(y_pred) - (1.0 - y_true) * tf.log(1.0 - y_pred))
    
    self.voicing_threshold = tf.Variable(0.15, trainable=False)
    tf.summary.scalar("model/voicing_threshold", self.voicing_threshold)

    self.loss = bkld(ref_probabilities, self.note_probabilities)
    self.est_notes = common.est_notes(self, args)
    self.training = common.optimizer(self, args)

    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     learning_rate = tf.train.exponential_decay(args.learning_rate, self.global_step, 30000, 0.9, True)
    #     tf.summary.scalar("model/learning_rate", learning_rate)
    #     self.training = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=self.global_step)


# cqt
HOP_LENGTH = 512

def parse_args(argv):
    parser = common.common_arguments_parser()
    parser.add_argument("--capacity_multiplier", type=int, help="Capacity multiplier")
    parser.add_argument("--spectrogram", type=str, help="Spectrogram method")
    parser.add_argument("--spectrogram_top_db", type=float, help="Spectrogram top_db")
    parser.add_argument("--spectrogram_filter_scale", type=float, help="Spectrogram filter_scale")
    parser.add_argument("--spectrogram_undertone_stacking", type=int, help="spectrogram undertone stacking")
    parser.add_argument("--spectrogram_overtone_stacking", type=int, help="spectrogram overtone stacking")
    parser.add_argument("--undertone_stacking", type=int, help="Undertone stacking in the model")
    parser.add_argument("--overtone_stacking", type=int, help="Overtone stacking in the model")
    args = parser.parse_args(argv)
    defaults = {
        "samplerate": 44100, "context_width": 0, "annotations_per_window": 50, "hop_size": 1,
        "frame_width": HOP_LENGTH,
        "note_range": 72, "min_note": 24,
        "evaluate_every": 5000,
        "evaluate_small_every": 1000,
        "spectrogram": "cqt",
        "learning_rate": 0.001,
        "learning_rate_decay": 0.85,
        "learning_rate_decay_steps": 5000,
        "undertone_stacking": 0,
        "overtone_stacking": 1,
        "spectrogram_undertone_stacking": 1,
        "spectrogram_overtone_stacking": 5,
        "spectrogram_top_db": 80,
        "spectrogram_filter_scale": 1.0,
        "capacity_multiplier": 64,

    }
    specified_args = common.argument_defaults(args, defaults)
    common.name(args, specified_args, "bittner")

    return args

def construct(args):
    network = NetworkMelody(args)

    with network.session.graph.as_default():
        spectrogram_function, spectrogram_thumb, spectrogram_info = common.spectrograms(args)
        # save spectrogram_thumb to hyperparams
        args.spectrogram_thumb = spectrogram_thumb

        hop_samples = args.frame_width*args.samplerate/44100
        print("hop_samples", hop_samples)
        def preload_fn(aa):
            aa.annotation = datasets.Annotation.from_time_series(*aa.annotation, hop_samples=hop_samples)
            aa.audio.load_resampled_audio(args.samplerate).load_spectrogram(spectrogram_function, spectrogram_thumb, spectrogram_info[2])

        def dataset_transform(tf_dataset, dataset):
            return tf_dataset.map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size_evaluation).prefetch(10)

        def dataset_transform_train(tf_dataset, dataset):
            return tf_dataset.shuffle(10**5).map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size).prefetch(10)

        valid_hooks = [MetricsHook(), VisualOutputHook(False, False, True, True), SaveBestModelHook(args.logdir), CSVOutputWriterHook(), AdjustVoicingHook()]
        train_dataset, test_datasets, validation_datasets = common.prepare_datasets(args.datasets, args, preload_fn, dataset_transform, dataset_transform_train, valid_hooks=valid_hooks)

        network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes, spectrogram_info=spectrogram_info)

    return network, train_dataset, validation_datasets, test_datasets


if __name__ == "__main__":
    common.main(sys.argv[1:], construct, parse_args)
