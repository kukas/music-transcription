import tensorflow as tf
import numpy as np

import datasets
from model import NetworkMelody
from collections import namedtuple
import sys
import common

import librosa

# cqt
HOP_LENGTH = 256
HARMONICS = [0.5, 1, 2, 3, 4, 5]
FMIN = 32.7
BINS_PER_OCTAVE = 60
N_BINS = BINS_PER_OCTAVE*6

def create_model(self, args):
    self.spectrogram.set_shape([None, len(HARMONICS), N_BINS, args.annotations_per_window + 2*args.context_width/HOP_LENGTH])

    spectrogram = self.spectrogram
    spectrogram = tf.transpose(spectrogram, [0, 3, 2, 1])

    print(spectrogram.shape)
    layer = spectrogram

    # layer = tf.pad(layer, ((0, 0), (0, 0), (41, 41), (0, 0)))
    print(layer.shape)

    context_size = int(args.context_width/HOP_LENGTH)

    with tf.name_scope('model_pitch'):
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.layers.conv2d(layer, 64, (5, 5), (1, 1), "same", activation=tf.nn.relu, use_bias=False)
        layer = tf.layers.dropout(layer, 0.25, training=self.is_training)
        residual = layer
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.layers.conv2d(layer, 64, (5, 5), (1, 1), "same", activation=tf.nn.relu, use_bias=False)
        layer = tf.layers.dropout(layer, 0.25, training=self.is_training)
        residual += layer
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.layers.conv2d(layer, 64, (9, 3), (1, 1), "same", activation=tf.nn.relu, use_bias=False)
        layer = tf.layers.dropout(layer, 0.25, training=self.is_training)
        residual += layer
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.layers.conv2d(layer, 64, (9, 3), (1, 1), "same", activation=tf.nn.relu, use_bias=False)
        layer = tf.layers.dropout(layer, 0.25, training=self.is_training)
        residual += layer
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.layers.conv2d(layer, 64, (5, 70), (1, 1), "same", activation=tf.nn.relu, use_bias=False)
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

    with tf.name_scope('model_voicing'):
        voicing_input = tf.concat([tf.stop_gradient(layer), spectrogram], axis=-1)
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

    self.loss = common.loss(self, args)
    self.est_notes = common.est_notes(self, args)
    self.training = common.optimizer(self, args)


def parse_args(argv):
    parser = common.common_arguments({
        "samplerate": 22050, "context_width": 14*HOP_LENGTH, "annotations_per_window": 20, "hop_size": 1, "frame_width": HOP_LENGTH,
        "note_range": 72, "min_note": 24,
        "evaluate_every": 5000,
        "evaluate_small_every": 1000,
    })
    # Model specific arguments
    parser.add_argument("--spectrogram", default="cqt", type=str, help="Postprocessing layer")

    args = parser.parse_args(argv)

    common.name(args, "cqt_voicing_residual_batchnorm")

    return args


def construct(args):
    network = NetworkMelody(args)

    with network.session.graph.as_default():
        def spec_function(audio, samplerate):

            cqt_list = []
            shapes = []
            for h in HARMONICS:
                cqt = librosa.cqt(
                    audio, sr=samplerate, hop_length=HOP_LENGTH, fmin=FMIN*float(h),
                    n_bins=N_BINS,
                    bins_per_octave=BINS_PER_OCTAVE
                )
                cqt_list.append(cqt)
                shapes.append(cqt.shape)

            shapes_equal = [s == shapes[0] for s in shapes]
            if not all(shapes_equal):
                print("NOT ALL", shapes_equal)
                min_time = np.min([s[1] for s in shapes])
                new_cqt_list = []
                for i in range(len(cqt_list)):
                    new_cqt_list.append(cqt_list[i][:, :min_time])
                cqt_list = new_cqt_list

            log_hcqt = ((1.0/80.0) * librosa.core.amplitude_to_db(
                np.abs(np.array(cqt_list)), ref=np.max)) + 1.0

            return (log_hcqt*65535).astype(np.uint16)

        spectrogram_thumb = "hcqt-fmin{}-oct{}-octbins{}-hop{}-db-uint16".format(FMIN, N_BINS/BINS_PER_OCTAVE, BINS_PER_OCTAVE, HOP_LENGTH)

        def preload_fn(aa):
            aa.annotation = datasets.Annotation.from_time_series(*aa.annotation, 512)
            aa.audio.load_resampled_audio(args.samplerate).load_spectrogram(spec_function, spectrogram_thumb, HOP_LENGTH)

        def dataset_transform(tf_dataset, dataset):
            return tf_dataset.map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size_evaluation).prefetch(10)

        def dataset_transform_train(tf_dataset, dataset):
            return tf_dataset.shuffle(10**5).map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size).prefetch(10)

        train_dataset, test_datasets, validation_datasets = common.prepare_datasets(args.datasets, args, preload_fn, dataset_transform, dataset_transform_train)

        network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes)

    return network, train_dataset, validation_datasets, test_datasets


if __name__ == "__main__":
    common.main(sys.argv[1:], construct, parse_args)
