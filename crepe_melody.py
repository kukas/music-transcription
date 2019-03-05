import tensorflow as tf
import numpy as np

import datasets
from model import NetworkMelody
from collections import namedtuple
import sys
import common

def create_model(self, args):
    # Get the melody annotation
    # round the annotations
    # self.annotations = tf.cast(tf.round(self.annotations), tf.int32)
    annotations = self.annotations[:, :, 0]
    window = self.window[:, :-1]

    if args.input_normalization:
        mean, var = tf.nn.moments(window, axes=[1])
        mean = tf.expand_dims(mean, axis=1)

        epsilon = 1e-3
        std_inv = tf.math.rsqrt(var + epsilon)
        std_inv = tf.expand_dims(std_inv, axis=1)

        window = (window - mean) * std_inv

    window_with_channel = tf.expand_dims(window, axis=2)

    voicing_ref = tf.cast(tf.greater(annotations, 0), tf.float32)

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
            first_layer.append(common.bn_conv(window_with_channel[:,512*0:512*1,:], 32*capacity_multiplier, 512, 64, "valid", activation=tf.nn.relu, reuse=None, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:,512*1:512*2,:], 32*capacity_multiplier, 512, 32, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:,512*2:512*3,:], 32*capacity_multiplier, 512, 32, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:,512*3:512*4,:], 32*capacity_multiplier, 512, 16, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:,512*4:512*5,:], 32*capacity_multiplier, 512, 16, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:,512*5:512*6,:], 32*capacity_multiplier, 512, 8, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:,512*6:512*7,:], 32*capacity_multiplier, 512, 8, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            
            first_layer.append(common.bn_conv(window_with_channel[:,512*7:512*9,:], 32*capacity_multiplier, 512, 4, "same", activation=tf.nn.relu, reuse=True, training=self.is_training))
            
            first_layer.append(common.bn_conv(window_with_channel[:,512*9:512*10,:], 32*capacity_multiplier, 512, 8, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:,512*10:512*11,:], 32*capacity_multiplier, 512, 8, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:,512*11:512*12,:], 32*capacity_multiplier, 512, 16, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:,512*12:512*13,:], 32*capacity_multiplier, 512, 16, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:,512*13:512*14,:], 32*capacity_multiplier, 512, 32, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:,512*14:512*15,:], 32*capacity_multiplier, 512, 32, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
            first_layer.append(common.bn_conv(window_with_channel[:,512*15:512*16,:], 32*capacity_multiplier, 512, 64, "valid", activation=tf.nn.relu, reuse=True, training=self.is_training))
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

    assert output_layer.shape.as_list() == [None, self.bin_count]

    # dense = tf.layers.dense(window, 1024, activation=tf.nn.relu)
    # output_layer = tf.layers.dense(dense, self.bin_count, activation=None, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)

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

    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.loss += args.l2_loss_weight * tf.reduce_sum(reg_variables)

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
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--batch_size", default=32, type=int, help="Number of examples in one batch")
    parser.add_argument("--annotations_per_window", default=1, type=int, help="Number of annotations in one example.")
    parser.add_argument("--frame_width", default=round(256/(44100/16000)), type=int, help="Number of samples per annotation = hop size.")
    parser.add_argument("--context_width", default=int(np.ceil((2048-93)/2)), type=int, help="Number of context samples on both sides of the example window.")
    parser.add_argument("--note_range", default=128, type=int, help="Note range.")
    parser.add_argument("--samplerate", default=16000, type=int, help="Audio samplerate used in the model, resampling is done automatically.")
    parser.add_argument("--logdir", default=None, type=str, help="Path to model directory.")
    parser.add_argument("--checkpoint", default="model", type=str, help="Checkpoint name.")
    parser.add_argument("--evaluate", action='store_true', help="Only evaluate.")
    parser.add_argument("--full_trace", action='store_true', help="Profile Tensorflow session.")
    parser.add_argument("--debug_memory_leaks", action='store_true', help="Debug memory leaks.")
    parser.add_argument("--cpu", action='store_true', help="Disable GPU.")
    # Model specific arguments
    parser.add_argument("--input_normalization", action='store_true', default=True, help="Enable normalizing each input example")
    parser.add_argument("--no_input_normalization", action='store_true', dest='input_normalization', help="Disable normalizing each input example")
    parser.add_argument("--learning_rate", default=0.0002, type=float, help="Learning rate")
    parser.add_argument("--capacity_multiplier", default=16, type=int, help="Capacity multiplier of the model")
    parser.add_argument("--clip_gradients", default=0.0, type=float, help="Clip gradients by global norm")
    parser.add_argument("--l2_loss_weight", default=0.0, type=float, help="L2 loss weight")
    parser.add_argument("--multiresolution_convolution", default=0, type=int, help="Number of different resolution of the first convolution layer")
    parser.add_argument("--bins_per_semitone", default=1, type=int, help="Bins per semitone")
    parser.add_argument("--annotation_smoothing", default=0.0, type=float, help="Set standard deviation of the gaussian blur for the frame annotations")
    parser.add_argument("--miss_weight", default=1.0, type=float, help="Weight for missed frames in the loss function")
    parser.add_argument("--variable_stride", action='store_true', default=False, help="Variable stride")
    parser.add_argument("--first_layer_capacity", default=1, type=int, help="Capacity multiplier")

    args = parser.parse_args(argv)

    common.name(args, "crepe")

    return args

def construct(args):
    network = NetworkMelody(args)

    with network.session.graph.as_default():
        def preload_fn(aa): return aa.audio.load_resampled_audio(args.samplerate)

        def dataset_transform(tf_dataset, dataset):
            return tf_dataset.map(dataset.prepare_example, num_parallel_calls=4).batch(args.batch_size).prefetch(1)

        def dataset_transform_train(tf_dataset, dataset):
            return tf_dataset.shuffle(10**5).map(dataset.prepare_example, num_parallel_calls=4).batch(args.batch_size).prefetch(1)

        train_dataset, test_datasets, validation_datasets = common.prepare_datasets(args.datasets, args, preload_fn, dataset_transform, dataset_transform_train)

        network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes)

    return network, train_dataset, validation_datasets, test_datasets


def main(argv):
    print(argv)
    args = parse_args(argv)
    # Construct the network
    network, train_dataset, validation_datasets, test_datasets = construct(args)
    
    if args.evaluate:
        for vd in test_datasets:
            print("{} evaluation".format(vd.name))
            network.evaluate(vd)
    else:
        network.train(train_dataset, args.epochs, validation_datasets, save_every_n_batches=10000)
        network.save()

if __name__ == "__main__":
    main(sys.argv[1:])
