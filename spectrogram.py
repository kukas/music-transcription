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
    self.spectrogram.set_shape([None, len(HARMONICS), N_BINS, args.annotations_per_window + 2*args.context_width/HOP_LENGTH])
    
    spectrogram = self.spectrogram
    spectrogram = tf.transpose(spectrogram, [0, 3, 2, 1])

    print(spectrogram.shape)
    layer = spectrogram

    # layer = tf.pad(layer, ((0, 0), (0, 0), (41, 41), (0, 0)))
    print(layer.shape)

    with tf.name_scope('model'):
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.layers.conv2d(layer, 64, (5, 5), (1, 1), "same", activation=tf.nn.relu, use_bias=False)
        residual = layer
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.layers.conv2d(layer, 64, (5, 5), (1, 1), "same", activation=tf.nn.relu, use_bias=False)
        residual += layer
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.layers.conv2d(layer, 64, (3, 3), (1, 1), "same", activation=tf.nn.relu, use_bias=False)
        residual += layer
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.layers.conv2d(layer, 64, (3, 3), (1, 1), "same", activation=tf.nn.relu, use_bias=False)
        residual += layer
        # layer3 = tf.layers.conv2d(layer_1, 64, (3, 1), (1, 1), "same", activation=tf.nn.relu, use_bias=False, dilation_rate=2)
        # layer3 = tf.layers.conv2d(layer_3, 64, (3, 1), (1, 1), "same", activation=tf.nn.relu, use_bias=False, dilation_rate=4)
        # layer3 = tf.layers.conv2d(layer_3, 64, (3, 1), (1, 1), "same", activation=tf.nn.relu, use_bias=False, dilation_rate=8)
        # layer3 = tf.layers.conv2d(layer_3, 64, (3, 1), (1, 1), "same", activation=tf.nn.relu, use_bias=False, dilation_rate=16)
        # conv_gate = tf.layers.conv2d(layer_3)
        # layer3_out = layer3 * tf.sigmoid(conv_gate)

        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = tf.layers.conv2d(layer, 64, (3, 70), (1, 1), "same", activation=tf.nn.relu, use_bias=False)
        residual += layer

        print(residual.shape)
        residual = residual[:, 7:-7, :, :]
        print(residual.shape)

        layer = tf.layers.batch_normalization(residual, training=self.is_training)
        layer = tf.layers.conv2d(layer, 1, (1, 1), (1, 1), "same", activation=None, use_bias=False)

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
        "samplerate": 22050, "context_width": 7*HOP_LENGTH, "annotations_per_window": 10, "hop_size": 10, "frame_width": HOP_LENGTH,
        "note_range": 72, "min_note": 24
    })
    # Model specific arguments
    parser.add_argument("--spectrogram", default="cqt", type=str, help="Postprocessing layer")

    args = parser.parse_args(argv)

    common.name(args, "cqt_resid")

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

            # cqt = librosa.core.cqt(audio, samplerate, hop_length=HOP_LENGTH, fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE)
            # # log scaling
            # cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            # # uint8 compression
            # cqt = ((cqt/80+1)*255).astype(np.uint8)
            # return cqt

        spectrogram_thumb = "hcqt-fmin{}-oct{}-octbins{}-hop{}-db-uint16".format(FMIN, N_BINS/BINS_PER_OCTAVE, BINS_PER_OCTAVE, HOP_LENGTH)

        def preload_fn(aa):
            aa.annotation.resample(256/22050)
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
