import tensorflow as tf
import numpy as np

import datasets
from model import NetworkMelody, VD
from collections import namedtuple

import common

def create_model(self, args):
    # Get the melody annotation
    annotations = self.annotations[:, :, 0]
    window = self.window[:, :-1]

    if args["input_normalization"]:
        mean, var = tf.nn.moments(window, axes=[1])
        mean = tf.expand_dims(mean, axis=1)

        epsilon = 1e-3
        std_inv = tf.math.rsqrt(var + epsilon)
        std_inv = tf.expand_dims(std_inv, axis=1)

        window = (window - mean) * std_inv

    window_with_channel = tf.expand_dims(window, axis=2)

    voicing_ref = tf.cast(tf.greater(annotations, 0), tf.float32)

    capacity_multiplier = args["capacity_multiplier"]

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
    output_layer = tf.layers.dense(audio_net, self.note_range, activation=None, bias_regularizer=tf.nn.l2_loss, kernel_regularizer=tf.nn.l2_loss)

    assert output_layer.shape.as_list() == [None, self.note_range]

    # dense = tf.layers.dense(window, 1024, activation=tf.nn.relu)
    # output_layer = tf.layers.dense(dense, self.note_range, activation=None)

    self.note_logits = tf.reshape(output_layer, [-1, self.annotations_per_window, self.note_range])
    self.note_probabilities = tf.nn.softmax(self.note_logits)
    self.est_notes = tf.argmax(self.note_logits, axis=2, output_type=tf.int32)

    # Possible only when self.annotations_per_window == 1
    assert self.annotations_per_window == 1
    flat_annotations = tf.reshape(annotations, [-1])
    flat_note_logits = output_layer
    flat_voicing_ref = tf.reshape(voicing_ref, [-1])
    print(flat_annotations.shape, flat_note_logits.shape, flat_voicing_ref.shape)

    self.loss = tf.losses.sparse_softmax_cross_entropy(flat_annotations, flat_note_logits, weights=flat_voicing_ref)
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.loss += 0.001 * tf.reduce_sum(reg_variables)

    # self.training = tf.train.AdamOptimizer(0.0002).minimize(self.loss, global_step=self.global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):                                                            
        optimizer = tf.train.AdamOptimizer(args["learning_rate"])
        # Get the gradient pairs (Tensor, Variable)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        grads, tvars = zip(*grads_and_vars)
        grads, _ = tf.clip_by_global_norm(grads, 3.0)

        grads_and_vars = list(zip(grads, tvars))
        # Update the weights wrt to the gradient
        self.training = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # Save the grads and vars with tf.summary.histogram
        for grad, var in grads_and_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
            tf.summary.histogram(var.name, var)



args = {
    "threads": 6,
    "batch_size": 32,
    "annotations_per_window": 1,
    "frame_width": round(256/(44100/16000)),  # frame_width of MedleyDB resampled to 16000 Hz
    "context_width": int(np.ceil((2048-93)/2)),
    "note_range": 128,
    "samplerate": 16000,
    "input_normalization": True,
    "learning_rate": 0.0002,
    "capacity_multiplier": 4,
}

# To restore model from a checkpoint
# args["logdir"] = "models/crepe_4mult_normalized-01-25_181717-bs32-apw1-fw93-ctx978-nr128-sr16000"

name_normalized = ("_normalized" if args["input_normalization"] else "")
common.name(args, "crepe_{}mult{}".format(args["capacity_multiplier"], name_normalized))

# Construct the network
network = NetworkMelody(threads=args["threads"])

with network.session.graph.as_default():
    sess = network.session

    def preload_fn(aa): return aa.audio.load_resampled_audio(args["samplerate"])

    def dataset_transform(dataset):
        return dataset.batch(128).prefetch(1)

    def dataset_transform_train(dataset):
        return dataset.shuffle(20000).batch(args["batch_size"]).prefetch(1)

    wjazzd_train, wjazzd_validation, wjazzd_small_validation = datasets.wjazzd.prepare(preload_fn)
    wjazzd_validation_dataset = datasets.AADataset(wjazzd_validation, args, dataset_transform)
    wjazzd_small_validation_dataset = datasets.AADataset(wjazzd_small_validation, args, dataset_transform)

    medleydb_train, medleydb_validation, medleydb_small_validation = datasets.medleydb.prepare(preload_fn)
    medleydb_validation_dataset = datasets.AADataset(medleydb_validation, args, dataset_transform)
    medleydb_small_validation_dataset = datasets.AADataset(medleydb_small_validation, args, dataset_transform)

    mdb_melody_synth_train, mdb_melody_synth_validation, _ = datasets.mdb_melody_synth.prepare(preload_fn)
    mdb_melody_synth_validation_dataset = datasets.AADataset(mdb_melody_synth_validation, args, dataset_transform)

    mdb_stem_synth_train, _, mdb_stem_synth_small_validation = datasets.mdb_stem_synth.prepare(preload_fn)
    mdb_stem_synth_small_validation_dataset = datasets.AADataset(mdb_stem_synth_small_validation, args, dataset_transform)

    _, _, mdb_mf0_synth_small_validation = datasets.mdb_mf0_synth.prepare(preload_fn)
    mdb_mf0_synth_small_validation_dataset = datasets.AADataset(mdb_mf0_synth_small_validation, args, dataset_transform)

    train_dataset = datasets.AADataset(medleydb_train+wjazzd_train+mdb_stem_synth_train+mdb_melody_synth_train, args, dataset_transform_train)

    network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes, dataset_preload_fn=preload_fn, dataset_transform=dataset_transform)

    validation_datasets = [
        VD(datasets.mdb_mf0_synth.prefix+"_small", mdb_mf0_synth_small_validation_dataset, 3000, True),
        VD(datasets.mdb_stem_synth.prefix+"_small", mdb_stem_synth_small_validation_dataset, 3000, True),
        VD(datasets.medleydb.prefix+"_small", medleydb_small_validation_dataset, 3000, True),
        VD(datasets.medleydb.prefix, medleydb_validation_dataset, 20000, False),
        VD(datasets.mdb_melody_synth.prefix, mdb_melody_synth_validation_dataset, 30000, False),
        VD(datasets.wjazzd.prefix, wjazzd_validation_dataset, 30000, False),
    ]

epochs = 3

network.train(train_dataset, epochs, validation_datasets, save_every_n_batches=10000)
network.save()

# print("ORCHSET evaluation")
# valid_data_orchset = datasets.orchset.dataset("data/Orchset/")
# orchset_dataset = datasets.AADataset(valid_data_orchset, args, dataset_transform)
# network.evaluate(orchset_dataset, print_detailed=True)
