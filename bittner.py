import tensorflow as tf

import numpy as np

import datasets
from model import NetworkMelody
from collections import namedtuple
import sys
import common

import librosa
import evaluation

from model import VD, VisualOutputHook, MetricsHook, SaveBestModelHook, EvaluationHook

def create_model(self, args):
    spectrogram = self.spectrogram

    with tf.name_scope('model'):
        layer = spectrogram
        layer = tf.layers.batch_normalization(spectrogram, training=self.is_training)
        layer = tf.layers.conv2d(layer, 128, (5, 5), (1, 1), "same", activation=tf.nn.relu)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.layers.conv2d(layer, 64, (5, 5), (1, 1), "same", activation=tf.nn.relu)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.layers.conv2d(layer, 64, (3, 3), (1, 1), "same", activation=tf.nn.relu)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.layers.conv2d(layer, 64, (3, 3), (1, 1), "same", activation=tf.nn.relu)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.layers.conv2d(layer, 8, (3, 70), (1, 1), "same", activation=tf.nn.relu)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.layers.conv2d(layer, 1, (1, 1), (1, 1), "same", activation=tf.nn.sigmoid)

        note_output = tf.squeeze(layer, -1)
        print(note_output.shape)
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

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        learning_rate = tf.train.exponential_decay(args.learning_rate, self.global_step, 30000, 0.9, True)
        tf.summary.scalar("model/learning_rate", learning_rate)
        self.training = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=self.global_step)


# cqt
HOP_LENGTH = 512

def parse_args(argv):
    parser = common.common_arguments({
        "samplerate": 44100, "context_width": 0, "annotations_per_window": 50, "hop_size": 1,
        "frame_width": HOP_LENGTH,
        "note_range": 72, "min_note": 24,
        "evaluate_every": 5000,
        "evaluate_small_every": 1000,
    })
    # Model specific arguments
    parser.add_argument("--spectrogram", default="hcqt", type=str, help="Postprocessing layer")

    args = parser.parse_args(argv)

    common.name(args, "bittner")

    return args


class AdjustVoicingHook(EvaluationHook):
    def before_run(self, ctx, vd):
        return [ctx.est_notes_confidence]

    def after_run(self, ctx, vd, additional):
        print("Adjusting voicing threshold")
        # np.save(vd.name+"_est_notes.npy", np.concatenate(list(additional[ctx.est_notes].values())) )
        # np.save(vd.name+"_est_notes_confidence.npy", np.concatenate(list(additional[ctx.est_notes_confidence].values())))
        # all_ref = np.concatenate([aa.annotation.freqs for aa in vd.dataset._annotated_audios])
        # np.save(vd.name+"_ref.npy", all_ref)

        thresholds = np.arange(0.0, 1.0, 0.01)
        results = []
        for threshold in thresholds:
            threshold_results = []
            for uid, est_notes_confidence in additional[ctx.est_notes_confidence].items():
                aa = vd.dataset.get_annotated_audio_by_uid(uid)
                ref_voicing = aa.annotation.voicing
                est_voicing = est_notes_confidence > threshold
                voicing_accuracy = evaluation.melody.voicing_accuracy(ref_voicing, est_voicing)
                threshold_results.append(voicing_accuracy)
            results.append(np.mean(threshold_results))

        best_threshold = thresholds[np.argmax(results)]
        print("New voicing threshold {:.2f} {:.3f}".format(best_threshold, np.max(results)))
        ctx.voicing_threshold.load(best_threshold, ctx.session)

def construct(args):
    network = NetworkMelody(args)

    with network.session.graph.as_default():
        spectrogram_function, spectrogram_thumb, spectrogram_info = common.spectrograms(args)

        def preload_fn(aa):
            aa.annotation = datasets.Annotation.from_time_series(*aa.annotation, 512)
            aa.audio.load_resampled_audio(args.samplerate).load_spectrogram(spectrogram_function, spectrogram_thumb, HOP_LENGTH)

        def dataset_transform(tf_dataset, dataset):
            return tf_dataset.map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size_evaluation).prefetch(10)

        def dataset_transform_train(tf_dataset, dataset):
            return tf_dataset.shuffle(10**5).map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size).prefetch(10)

        valid_hooks = [MetricsHook(write_estimations=True), VisualOutputHook(False, False, True, True), SaveBestModelHook(args.logdir), AdjustVoicingHook()]
        train_dataset, test_datasets, validation_datasets = common.prepare_datasets(args.datasets, args, preload_fn, dataset_transform, dataset_transform_train, valid_hooks=valid_hooks)

        network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes, spectrogram_info=spectrogram_info)

    return network, train_dataset, validation_datasets, test_datasets


if __name__ == "__main__":
    common.main(sys.argv[1:], construct, parse_args)
