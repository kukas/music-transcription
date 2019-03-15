import tensorflow as tf
from tensorflow.keras import layers
# from tensorflow.keras import backend as K


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
    # window = self.window
    # window = common.input_normalization(window, args)
    # window_with_channel = tf.expand_dims(window, axis=2)

    layer_confs = [
        (128, (5, 5), tf.nn.relu),
        (64, (5, 5), tf.nn.relu),
        (64, (3, 3), tf.nn.relu),
        (64, (3, 3), tf.nn.relu),
        (8, (3, 70), tf.nn.relu),
        (1, (1, 1), None),
    ]

    self.spectrogram.set_shape([None, N_BINS, args.annotations_per_window, 1])
    spectrogram = self.spectrogram
    spectrogram = tf.transpose(spectrogram, [0, 2, 1, 3])

    total_bins = args.note_range * args.bins_per_semitone
    bottom_padding = int(librosa.core.hz_to_midi(FMIN) * args.bins_per_semitone)
    top_padding = int(total_bins - N_BINS - bottom_padding)

    print("paddings:", bottom_padding, top_padding)

    spectrogram = tf.pad(spectrogram, ((0, 0), (0, 0), (bottom_padding, top_padding), (0, 0)))
    print(spectrogram.shape)
    layer = spectrogram

    # tf.summary.image("spectrogram", spectrogram)
    # layer = layer + 0.001*layer*tf.sigmoid(tf.Variable(1.0))
    # layer = tf.layers.conv2d(layer, 1, (1, 1), (1, 1), "same", activation=None)

    with tf.name_scope('model'):
        for filters, kernel_size, activation in layer_confs:
            layer = tf.layers.conv2d(layer, filters, kernel_size, (1,1), "same", activation=None, use_bias=True)
            # layer = tf.layers.batch_normalization(layer, training=self.is_training)
            if activation is not None:
                layer = activation(layer)

    print(layer.shape)
    output_layer = tf.squeeze(layer, -1)
    print(output_layer.shape)

    # if output_layer.shape.as_list() != [None, self.annotations_per_window, self.bin_count]:
    #     print("shape not compatible, adding FC layer")
    #     output_layer = tf.nn.relu(output_layer)
    #     output_layer = tf.layers.flatten(output_layer)
    #     output_layer = tf.layers.dense(output_layer, self.annotations_per_window*self.bin_count, activation=None, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
    #     output_layer = tf.reshape(output_layer, [-1, self.annotations_per_window, self.bin_count])

    self.note_logits = output_layer

    self.loss = common.loss(self, args)
    self.est_notes = common.est_notes(self, args)
    self.training = common.optimizer(self, args)


def parse_args(argv):
    parser = common.common_arguments({
        "samplerate": 44100, "context_width": 0, "annotations_per_window": 50, "hop_size": 10, "frame_width": 256
    })
    # Model specific arguments
    parser.add_argument("--spectrogram", default="cqt", type=str, help="Postprocessing layer")

    args = parser.parse_args(argv)

    common.name(args, "cqt_nomodel")

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

            return (log_hcqt*255).astype(np.uint8)

            # cqt = librosa.core.cqt(audio, samplerate, hop_length=HOP_LENGTH, fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE)
            # # log scaling
            # cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            # # uint8 compression
            # cqt = ((cqt/80+1)*255).astype(np.uint8)
            # return cqt

        spectrogram_thumb = "hcqt-fmin{}-oct{}-octbins{}-hop{}-db-uint8".format(FMIN, N_BINS/BINS_PER_OCTAVE, BINS_PER_OCTAVE, HOP_LENGTH)

        def preload_fn(aa):
            return aa.audio.load_resampled_audio(args.samplerate).load_spectrogram(spec_function, spectrogram_thumb, HOP_LENGTH)

        def dataset_transform(tf_dataset, dataset):
            return tf_dataset.map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size_evaluation).prefetch(10)

        def dataset_transform_train(tf_dataset, dataset):
            return tf_dataset.shuffle(10**5).map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size).prefetch(10)

        train_dataset, test_datasets, validation_datasets = common.prepare_datasets(args.datasets, args, preload_fn, dataset_transform, dataset_transform_train)

        network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes)

    return network, train_dataset, validation_datasets, test_datasets


if __name__ == "__main__":
    common.main(sys.argv[1:], construct, parse_args)
