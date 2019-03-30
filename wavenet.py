import tensorflow as tf
import numpy as np

import datasets
from model import NetworkMelody
from collections import namedtuple
import sys
import common

# https://github.com/lmartak/amt-wavenet/blob/master/wavenet/model.py


def create_model(self, args):
    window = self.window
    window = common.input_normalization(window, args)
    window_with_channel = tf.expand_dims(window, axis=2)

    initial_layer = tf.layers.conv1d(window_with_channel, args.residual_channels, args.initial_filter_width, 1, args.initial_filter_padding,
                                     activation=None, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)

    skip_connections = []
    dilations = [2**x for x in range(int(np.log2(args.max_dilation))+1)]*args.stack_number
    print(dilations)
    current_layer = initial_layer
    with tf.name_scope('dilated_stack'):
        for layer_index, dilation in enumerate(dilations):
            with tf.name_scope('layer{}'.format(layer_index)):
                conv_filter = tf.layers.conv1d(current_layer, args.residual_channels, args.filter_width, 1, "same", dilation_rate=dilation,
                                               use_bias=args.use_biases, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
                conv_gate = tf.layers.conv1d(current_layer, args.residual_channels, args.filter_width, 1, "same", dilation_rate=dilation,
                                             use_bias=args.use_biases, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
                out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)
                skip = tf.layers.conv1d(out, args.skip_channels, 1, 1, "same", use_bias=args.use_biases, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
                transformed = tf.layers.conv1d(out, args.residual_channels, 1, 1, "same", use_bias=args.use_biases, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
                if args.dilation_layer_dropout:
                    transformed = tf.layers.dropout(transformed, args.dilation_layer_dropout, training=self.is_training)
                current_layer = transformed + current_layer

                skip_connections.append(skip)
                print(skip)

    with tf.name_scope('postprocessing'):
        skip_sum = tf.math.add_n(skip_connections)
        skip = tf.nn.relu(skip_sum)
        if args.skip_layer_dropout:
            skip = tf.layers.dropout(skip, args.skip_layer_dropout, training=self.is_training)

        # skip = tf.layers.average_pooling1d(skip, 93, 93, "valid")
        # skip = tf.layers.conv1d(skip, self.bin_count, 3, 1, "same", activation=tf.nn.relu, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
        # output_layer = tf.layers.conv1d(skip, self.bin_count, 3, 1, "same", activation=None, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)

        output_layer = common.add_layers_from_string(skip, args.postprocessing)

        # skip = tf.layers.conv1d(skip, 256, 16, 8, "same", activation=tf.nn.relu, use_bias=args.use_biases, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
        # skip = tf.layers.conv1d(skip, 256, 16, 8, "same", activation=tf.nn.relu, use_bias=args.use_biases, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
        # skip = tf.nn.relu(skip_sum)
        print(output_layer.shape)

    if output_layer.shape.as_list() != [None, self.annotations_per_window, self.bin_count]:
        print("shape not compatible, adding FC layer")
        output_layer = tf.nn.relu(output_layer)
        output_layer = tf.layers.flatten(output_layer)
        output_layer = tf.layers.dense(output_layer, self.annotations_per_window*self.bin_count, activation=None, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
        output_layer = tf.reshape(output_layer, [-1, self.annotations_per_window, self.bin_count])

    self.note_logits = output_layer

    self.loss = common.loss(self, args)
    self.est_notes = common.est_notes(self, args)
    self.training = common.optimizer(self, args)


def parse_args(argv):
    parser = common.common_arguments({"context_width": 0})
    # Model specific arguments
    parser.add_argument("--initial_filter_width", default=32, type=int, help="First conv layer filter width")
    parser.add_argument("--initial_filter_padding", default="same", type=str, help="First conv layer padding")
    parser.add_argument("--filter_width", default=3, type=int, help="Dilation stack filter width (2 or 3)")
    parser.add_argument("--use_biases", action='store_true', default=False, help="Use biases in the convolutions")
    parser.add_argument("--skip_channels", default=64, type=int, help="Skip channels")
    parser.add_argument("--residual_channels", default=32, type=int, help="Residual channels")
    parser.add_argument("--stack_number", default=1, type=int, help="Number of dilated stacks")
    parser.add_argument("--max_dilation", default=512, type=int, help="Maximum dilation rate")
    parser.add_argument("--dilation_layer_dropout", default=0.0, type=float, help="Dropout in dilation layer")
    parser.add_argument("--skip_layer_dropout", default=0.0, type=float, help="Dropout in skip connections")
    parser.add_argument("--postprocessing", default="avgpool_p93_s93_Psame->conv_f256_k16_s8_Psame_arelu->conv_f256_k16_s8_Psame_arelu", type=str, help="Postprocessing layer")

    args = parser.parse_args(argv)

    common.name(args, "wavenet")

    return args


def construct(args):
    network = NetworkMelody(args)

    with network.session.graph.as_default():
        def preload_fn(aa):
            aa.annotation = datasets.Annotation.from_time_series(*aa.annotation)
            aa.audio.load_resampled_audio(args.samplerate)

        def dataset_transform(tf_dataset, dataset):
            return tf_dataset.map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size_evaluation).prefetch(10)

        def dataset_transform_train(tf_dataset, dataset):
            return tf_dataset.shuffle(10**5).map(dataset.prepare_example, num_parallel_calls=args.threads).filter(dataset.is_example_voiced).batch(args.batch_size).prefetch(10)

        train_dataset, test_datasets, validation_datasets = common.prepare_datasets(args.datasets, args, preload_fn, dataset_transform, dataset_transform_train)

        network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes)

    return network, train_dataset, validation_datasets, test_datasets


if __name__ == "__main__":
    common.main(sys.argv[1:], construct, parse_args)
