import tensorflow as tf
import numpy as np

import datasets
from model import NetworkMelody
from collections import namedtuple
import sys
import common

# https://github.com/lmartak/amt-wavenet/blob/master/wavenet/model.py


def create_model(self, args):
    receptive_field = args.stack_number * (args.max_dilation * 2) - (args.stack_number - 1)
    receptive_field_ms = (receptive_field * 1000) / args.samplerate

    context_width = self.context_width

    print("receptive field: {} samples, {:.4f} ms".format(receptive_field, receptive_field_ms))
    if self.context_width > receptive_field:
        context_width = receptive_field
        diff = self.context_width - receptive_field
        window = self.window[:, diff:-diff]
        print("cutting window {}->{}".format(self.window.shape, window.shape))
    else:
        window = self.window
        print("warning: receptive field larger than context width")
    
    window = common.input_normalization(window, args)
    window_with_channel = tf.expand_dims(window, axis=2)

    initial_layer = window_with_channel
    if args.initial_filter_width > 0:
        initial_layer = tf.layers.conv1d(initial_layer, args.residual_channels, args.initial_filter_width, 1, args.initial_filter_padding,
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
        if args.skip == "add":
            skip_sum = tf.math.add_n(skip_connections)
        elif args.skip == "concat":
            skip_sum = tf.concat(skip_connections, -1)
        elif args.skip == "last":
            skip_sum = skip_connections[-1]
        
        if context_width:
            skip_sum = skip_sum[:, context_width:-context_width, :]
        
        print("skip output", skip_sum.shape)
        skip = tf.nn.relu(skip_sum)
        if args.skip_layer_dropout:
            skip = tf.layers.dropout(skip, args.skip_layer_dropout, training=self.is_training)

        # skip = tf.layers.average_pooling1d(skip, 93, 93, "valid")
        # skip = tf.layers.conv1d(skip, self.bin_count, 3, 1, "same", activation=tf.nn.relu, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
        # output_layer = tf.layers.conv1d(skip, self.bin_count, 3, 1, "same", activation=None, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)

        output_layer = common.add_layers_from_string(self, skip, args.postprocessing)

        # skip = tf.layers.conv1d(skip, 256, 16, 8, "same", activation=tf.nn.relu, use_bias=args.use_biases, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
        # skip = tf.layers.conv1d(skip, 256, 16, 8, "same", activation=tf.nn.relu, use_bias=args.use_biases, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
        # skip = tf.nn.relu(skip_sum)
        print("after skip output processing", output_layer.shape)

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
    parser = common.common_arguments_parser()
    # Model specific arguments
    parser.add_argument("--use_biases", action='store_true', default=False, help="Use biases in the convolutions")

    parser.add_argument("--input_normalization", type=int, help="Enable normalizing each input example")
    parser.add_argument("--initial_filter_width", type=int, help="First conv layer filter width")
    parser.add_argument("--initial_filter_padding", type=str, help="First conv layer padding")
    parser.add_argument("--filter_width", type=int, help="Dilation stack filter width (2 or 3)")
    parser.add_argument("--skip_channels", type=int, help="Skip channels")
    parser.add_argument("--residual_channels", type=int, help="Residual channels")
    parser.add_argument("--stack_number", type=int, help="Number of dilated stacks")
    parser.add_argument("--max_dilation", type=int, help="Maximum dilation rate")
    parser.add_argument("--dilation_layer_dropout", type=float, help="Dropout in dilation layer")
    parser.add_argument("--skip_layer_dropout", type=float, help="Dropout in skip connections")
    parser.add_argument("--skip", type=str, help="Skip add or concat")
    parser.add_argument("--postprocessing", type=str, help="Postprocessing layer")

    args = parser.parse_args(argv)
    defaults = {
        "note_range": 72, "min_note": 24,
        "evaluate_every": 5000,
        "evaluate_small_every": 5000,
        "batch_size": 8,
        "annotations_per_window": 10,
        "context_width": 94,
        "annotation_smoothing": 0.18,
        "input_normalization": 1,
        "initial_filter_width": 32,
        "initial_filter_padding": "same",
        "filter_width": 3,
        "skip_channels": 64,
        "residual_channels": 32,
        "stack_number": 1,
        "max_dilation": 512,
        "dilation_layer_dropout": 0.0,
        "skip_layer_dropout": 0.0,
        "skip": "add",
        "postprocessing": "avgpool_p93_s93_Psame--conv_f256_k16_s8_Psame_arelu--conv_f256_k16_s8_Psame_arelu",
    }
    specified_args = common.argument_defaults(args, defaults)
    common.name(args, specified_args, "wavenet")

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
