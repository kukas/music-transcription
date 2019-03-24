import tensorflow as tf
import numpy as np

import datasets
from model import NetworkMelody
from collections import namedtuple
import sys
import common

def create_model(self, args):
    window = self.window[:, :-1]
    window = common.input_normalization(window, args)
    window_with_channel = tf.expand_dims(window, axis=2)

    capacity_multiplier = args.capacity_multiplier

    if args.multiresolution_convolution:
        first_layer = []

        for i in range(args.multiresolution_convolution):
            width = 2**(9-i)
            capacity = 32//args.multiresolution_convolution*args.first_layer_capacity
            l = common.bn_conv(window_with_channel, capacity*capacity_multiplier, width, 4, "same", activation=tf.nn.relu, training=self.is_training)
            print(l.shape, width)
            first_layer.append(l)

        audio_net = tf.concat(first_layer, 2)
    else:
        if args.variable_stride:
            first_layer = []
            # print(window_with_channel.shape)
            first_layer.append(common.bn_conv(window_with_channel[:, 512*0:512*1, :], 32*capacity_multiplier, 512, 64, "valid", activation=tf.nn.relu, reuse=None, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:, 512*1:512*2, :], 32*capacity_multiplier, 512, 32, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:, 512*2:512*3, :], 32*capacity_multiplier, 512, 32, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:, 512*3:512*4, :], 32*capacity_multiplier, 512, 16, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:, 512*4:512*5, :], 32*capacity_multiplier, 512, 16, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:, 512*5:512*6, :], 32*capacity_multiplier, 512, 8, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:, 512*6:512*7, :], 32*capacity_multiplier, 512, 8, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))

            first_layer.append(common.bn_conv(window_with_channel[:, 512*7:512*9, :], 32*capacity_multiplier, 512, 4, "same", activation=tf.nn.relu, reuse=True, training=self.is_training))

            first_layer.append(common.bn_conv(window_with_channel[:, 512*9:512*10, :], 32*capacity_multiplier, 512, 8, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:, 512*10:512*11, :], 32*capacity_multiplier, 512, 8, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:, 512*11:512*12, :], 32*capacity_multiplier, 512, 16, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:, 512*12:512*13, :], 32*capacity_multiplier, 512, 16, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:, 512*13:512*14, :], 32*capacity_multiplier, 512, 32, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:, 512*14:512*15, :], 32*capacity_multiplier, 512, 32, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:, 512*15:512*16, :], 32*capacity_multiplier, 512, 64, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            print(first_layer)
            audio_net = tf.concat(first_layer, 1)
        else:
            audio_net = common.bn_conv(window_with_channel, 32*capacity_multiplier, 512, 4, "same", activation=tf.nn.relu, training=self.is_training)

    audio_net = tf.layers.max_pooling1d(audio_net, 2, 2)
    audio_net = tf.layers.dropout(audio_net, 0.25, training=self.is_training)

    audio_net = common.bn_conv(audio_net, 4*capacity_multiplier, 64, 1, "same", activation=tf.nn.relu, training=self.is_training)
    audio_net = tf.layers.max_pooling1d(audio_net, 2, 2)
    audio_net = tf.layers.dropout(audio_net, 0.25, training=self.is_training)

    audio_net = common.bn_conv(audio_net, 4*capacity_multiplier, 64, 1, "same", activation=tf.nn.relu, training=self.is_training)
    audio_net = tf.layers.max_pooling1d(audio_net, 2, 2)
    audio_net = tf.layers.dropout(audio_net, 0.25, training=self.is_training)

    audio_net = common.bn_conv(audio_net, 4*capacity_multiplier, 64, 1, "same", activation=tf.nn.relu, training=self.is_training)
    audio_net = tf.layers.max_pooling1d(audio_net, 2, 2)
    audio_net = tf.layers.dropout(audio_net, 0.25, training=self.is_training)

    audio_net = common.bn_conv(audio_net, 8*capacity_multiplier, 64, 1, "same", activation=tf.nn.relu, training=self.is_training)
    audio_net = tf.layers.max_pooling1d(audio_net, 2, 2)
    audio_net = tf.layers.dropout(audio_net, 0.25, training=self.is_training)

    audio_net = common.bn_conv(audio_net, 16*capacity_multiplier, 64, 1, "same", activation=tf.nn.relu, training=self.is_training)
    audio_net = tf.layers.max_pooling1d(audio_net, 2, 2)
    audio_net = tf.layers.dropout(audio_net, 0.25, training=self.is_training)

    audio_net = tf.layers.flatten(audio_net)

    output_layer = tf.layers.dense(audio_net, self.bin_count, activation=None, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
    self.note_logits = tf.reshape(output_layer, [-1, self.annotations_per_window, self.bin_count])

    self.loss = common.loss(self, args)
    self.est_notes = common.est_notes(self, args)
    self.training = common.optimizer(self, args)

def parse_args(argv):
    parser = common.common_arguments({"context_width": 978})
    parser.add_argument("--capacity_multiplier", default=16, type=int, help="Capacity multiplier of the model")
    parser.add_argument("--multiresolution_convolution", default=0, type=int, help="Number of different resolution of the first convolution layer")
    parser.add_argument("--variable_stride", action='store_true', default=False, help="Variable stride")
    parser.add_argument("--first_layer_capacity", default=1, type=int, help="Capacity multiplier")

    args = parser.parse_args(argv)

    common.name(args, "crepe")

    return args


def construct(args):
    network = NetworkMelody(args)

    with network.session.graph.as_default():
        def preload_fn(aa):
            aa.annotation = datasets.Annotation.from_time_series(*aa.annotation)
            aa.audio.load_resampled_audio(args.samplerate)
        
        # augment_audio_basa = datasets.Audio("/mnt/tera/jirka/V1/MatthewEntwistle_FairerHopes/MatthewEntwistle_FairerHopes_STEMS/MatthewEntwistle_FairerHopes_STEM_07.wav",
        #                                     "augment_low").load_resampled_audio(args.samplerate).slice(20, 30)
        # augment_audio_perkuse = datasets.Audio("/mnt/tera/jirka/V1/MatthewEntwistle_FairerHopes/MatthewEntwistle_FairerHopes_STEMS/MatthewEntwistle_FairerHopes_STEM_08.wav",
        #                                        "augment_low").load_resampled_audio(args.samplerate).slice(20, 30)

        # augment_audio = augment_audio_basa.samples*10 + augment_audio_perkuse.samples*10

        def dataset_transform(tf_dataset, dataset):
            return tf_dataset.map(dataset.prepare_example).batch(args.batch_size_evaluation).prefetch(1)
            # return tf_dataset.map(dataset.prepare_example).map(dataset.mix_example_with(augment_audio)).batch(args.batch_size_evaluation).prefetch(1)

        def dataset_transform_train(tf_dataset, dataset):
            return tf_dataset.shuffle(10**5).map(dataset.prepare_example, num_parallel_calls=4).filter(dataset.is_example_voiced).batch(args.batch_size).prefetch(1)

        train_dataset, test_datasets, validation_datasets = common.prepare_datasets(args.datasets, args, preload_fn, dataset_transform, dataset_transform_train)

        network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes)

    return network, train_dataset, validation_datasets, test_datasets


if __name__ == "__main__":
    common.main(sys.argv[1:], construct, parse_args)
