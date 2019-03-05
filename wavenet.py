import tensorflow as tf
import numpy as np

import datasets
from model import NetworkMelody
from collections import namedtuple
import sys
import common

# https://github.com/lmartak/amt-wavenet/blob/master/wavenet/model.py
def create_model(self, args):
    # Get the melody annotation
    # round the annotations
    # self.annotations = tf.cast(tf.round(self.annotations), tf.int32)
    annotations = self.annotations[:, :, 0]
    # window = self.window[:, :-1]
    window = self.window

    if args.input_normalization:
        mean, var = tf.nn.moments(window, axes=[1])
        mean = tf.expand_dims(mean, axis=1)

        epsilon = 1e-3
        std_inv = tf.math.rsqrt(var + epsilon)
        std_inv = tf.expand_dims(std_inv, axis=1)

        window = (window - mean) * std_inv

    window_with_channel = tf.expand_dims(window, axis=2)

    voicing_ref = tf.cast(tf.greater(annotations, 0), tf.float32)

    initial_layer = tf.layers.conv1d(window_with_channel, args.residual_channels, args.initial_filter_width, 1, "same", activation=None, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)

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
                transformed = tf.layers.dropout(transformed, args.dilation_layer_dropout, training=self.is_training)
                current_layer = transformed + current_layer

                skip_connections.append(skip)
                print(skip)
    
    with tf.name_scope('postprocessing'):
        skip_sum = tf.math.add_n(skip_connections)
        skip = tf.nn.relu(skip_sum)
        skip = tf.layers.dropout(skip, args.skip_layer_dropout, training=self.is_training)
        skip = tf.layers.average_pooling1d(skip, 93, 93, "valid")
        # skip = tf.layers.conv1d(skip, 256, 16, 8, "same", activation=tf.nn.relu, use_bias=args.use_biases, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
        # skip = tf.layers.conv1d(skip, 256, 16, 8, "same", activation=tf.nn.relu, use_bias=args.use_biases, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
        # skip = tf.nn.relu(skip_sum)
        skip = tf.layers.conv1d(skip, 640, 3, 1, "same", activation=tf.nn.relu, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)
        output_layer = tf.layers.conv1d(skip, 640, 3, 1, "same", activation=None, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)

        print(skip.shape)

    # audio_net = tf.layers.flatten(skip)
    # audio_net = tf.layers.dropout(audio_net, args.skip_layer_dropout, training=self.is_training)

    # output_layer = tf.layers.dense(audio_net, self.annotations_per_window*self.bin_count, activation=None, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)

    # assert output_layer.shape.as_list() == [None, self.annotations_per_window*self.bin_count]

    self.note_logits = tf.reshape(output_layer, [-1, self.annotations_per_window, self.bin_count])

    batch_size = tf.shape(annotations)[0]

    note_bins = tf.range(0, 128, 1/self.bins_per_semitone, dtype=tf.float32)
    note_bins = tf.reshape(tf.tile(note_bins, [batch_size * self.annotations_per_window]), [batch_size, self.annotations_per_window, self.bin_count])

    peak_ref = self.annotations[:, :, 0]
    peak_ref = tf.cast(tf.abs(tf.tile(tf.reshape(peak_ref, [batch_size, self.annotations_per_window, 1]), [1, 1, self.bin_count]) - note_bins) < 0.5, tf.float32)

    if args.annotation_smoothing > 0:
        self.note_probabilities = tf.nn.sigmoid(self.note_logits)
        note_ref = tf.tile(tf.reshape(annotations, [-1, self.annotations_per_window, 1]), [1, 1, self.bin_count])
        ref_probabilities = tf.exp(-(note_ref-note_bins)**2/(args.annotation_smoothing**2))
        
        voicing_weights = tf.tile(tf.expand_dims(voicing_ref, -1), [1, 1, self.bin_count])
        # TODO přepsat hezčejc
        miss_weights = tf.ones_like(voicing_weights)*args.miss_weight + peak_ref*(1-args.miss_weight)
        self.loss = tf.losses.sigmoid_cross_entropy(ref_probabilities, self.note_logits, weights=voicing_weights*miss_weights)
    else:
        self.note_probabilities = tf.nn.softmax(self.note_logits)
        ref_bins = tf.cast(tf.round(self.annotations[:, 0] * self.bins_per_semitone), tf.int32)
        self.loss = tf.losses.sparse_softmax_cross_entropy(ref_bins, self.note_logits, weights=voicing_ref)

    peak_est = tf.cast(tf.argmax(self.note_logits, axis=2) / self.bins_per_semitone, tf.float32)
    peak_est = tf.cast(tf.abs(tf.tile(tf.reshape(peak_est, [batch_size, self.annotations_per_window, 1]), [1, 1, self.bin_count]) - note_bins) < 0.5, tf.float32)
    probs_around_peak = self.note_probabilities*peak_est
    self.est_notes = tf.reduce_sum(note_bins * probs_around_peak, axis=2)/tf.reduce_sum(probs_around_peak, axis=2)

    if args.l2_loss_weight > 0:
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_loss = tf.reduce_sum(tf.constant(args.l2_loss_weight)*reg_variables)
        tf.summary.scalar('metrics/train/l2_loss', l2_loss)
        tf.summary.scalar('metrics/train/ce_loss', self.loss)
        self.loss += l2_loss

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):                                                            
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        # Get the gradient pairs (Tensor, Variable)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        if args.clip_gradients > 0:
            grads, tvars = zip(*grads_and_vars)
            grads, _ = tf.clip_by_global_norm(grads, args.clip_gradients)
            grads_and_vars = list(zip(grads, tvars))
        # Update the weights wrt to the gradient
        self.training = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # Save the grads and vars with tf.summary.histogram
        for grad, var in grads_and_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
            tf.summary.histogram(var.name, var)

def parse_args(argv):
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=["mdb"], nargs="+", type=str, help="Datasets to use for this experiment")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train for.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--batch_size", default=32, type=int, help="Number of examples in one batch")
    parser.add_argument("--annotations_per_window", default=1, type=int, help="Number of annotations in one example.")
    parser.add_argument("--hop_size", default=None, type=int, help="Hop of the input window specified in number of annotations. Defaults to annotations_per_window")
    parser.add_argument("--frame_width", default=round(256/(44100/16000)), type=int, help="Number of samples per annotation = hop size.")
    parser.add_argument("--context_width", default=0, type=int, help="Number of context samples on both sides of the example window.")
    parser.add_argument("--note_range", default=128, type=int, help="Note range.")
    parser.add_argument("--samplerate", default=16000, type=int, help="Audio samplerate used in the model, resampling is done automatically.")
    parser.add_argument("--logdir", default=None, type=str, help="Path to model directory.")
    parser.add_argument("--checkpoint", default="model", type=str, help="Checkpoint name.")
    parser.add_argument("--evaluate", action='store_true', help="Evaluate after training. If an existing checkpoint is specified, it will be evaluated only.")
    parser.add_argument("--full_trace", action='store_true', help="Profile Tensorflow session.")
    parser.add_argument("--debug_memory_leaks", action='store_true', help="Debug memory leaks.")
    parser.add_argument("--cpu", action='store_true', help="Disable GPU.")
    # Model specific arguments
    parser.add_argument("--input_normalization", action='store_true', default=True, help="Enable normalizing each input example")
    parser.add_argument("--no_input_normalization", action='store_true', dest='input_normalization', help="Disable normalizing each input example")
    parser.add_argument("--learning_rate", default=0.0002, type=float, help="Learning rate")
    parser.add_argument("--clip_gradients", default=0.0, type=float, help="Clip gradients by global norm")
    parser.add_argument("--l2_loss_weight", default=0.0, type=float, help="L2 loss weight")
    parser.add_argument("--bins_per_semitone", default=1, type=int, help="Bins per semitone")
    parser.add_argument("--annotation_smoothing", default=0.0, type=float, help="Set standard deviation of the gaussian blur for the frame annotations")
    parser.add_argument("--miss_weight", default=1.0, type=float, help="Weight for missed frames in the loss function")
    parser.add_argument("--initial_filter_width", default=32, type=int, help="First conv layer filter width")
    parser.add_argument("--filter_width", default=3, type=int, help="Dilation stack filter width (2 or 3)")
    parser.add_argument("--use_biases", action='store_true', default=False, help="Use biases in the convolutions")
    parser.add_argument("--skip_channels", default=64, type=int, help="Skip channels")
    parser.add_argument("--residual_channels", default=32, type=int, help="Residual channels")
    parser.add_argument("--stack_number", default=1, type=int, help="Number of dilated stacks")
    parser.add_argument("--max_dilation", default=512, type=int, help="Maximum dilation rate")
    parser.add_argument("--dilation_layer_dropout", default=0.0, type=float, help="Dropout in dilation layer")
    parser.add_argument("--skip_layer_dropout", default=0.0, type=float, help="Dropout in skip connections")

    args = parser.parse_args(argv)

    common.name(args, "wavenet")

    return args

def construct(args):
    network = NetworkMelody(args)

    with network.session.graph.as_default():
        def preload_fn(aa):
            aa.audio.load_resampled_audio(args.samplerate)

        def dataset_transform(tf_dataset, dataset):
            return tf_dataset.map(dataset.prepare_example, num_parallel_calls=4).batch(args.batch_size).prefetch(1)

        def dataset_transform_train(tf_dataset, dataset):
            return tf_dataset.shuffle(10**5).map(dataset.prepare_example, num_parallel_calls=4).filter(dataset.is_example_voiced).batch(args.batch_size).prefetch(1)

        train_dataset, test_datasets, validation_datasets = common.prepare_datasets(args.datasets, args, preload_fn, dataset_transform, dataset_transform_train)

        network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes)

    return network, train_dataset, validation_datasets, test_datasets


def main(argv):
    print(argv)
    args = parse_args(argv)
    # Construct the network
    network, train_dataset, validation_datasets, test_datasets = construct(args)
    
    if not (args.evaluate and network.restored):
        try:
            network.train(train_dataset, args.epochs, validation_datasets, save_every_n_batches=10000)
            network.save()
        except KeyboardInterrupt:
            network.save()
            sys.exit()

    if args.evaluate:
        for vd in test_datasets:
            print("{} evaluation".format(vd.name))
            network.evaluate(vd)


if __name__ == "__main__":
    main(sys.argv[1:])
