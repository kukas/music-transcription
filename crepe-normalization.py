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

    capacity_multiplier = 4

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
    output_layer = tf.layers.dense(audio_net, self.note_range, activation=None)

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

    self.training = tf.train.AdamOptimizer(0.0002).minimize(self.loss, global_step=self.global_step)


args = {
    "threads": 6,
    "batch_size": 32,
    "annotations_per_window": 1,
    "frame_width": round(256/(44100/16000)),  # frame_width of MedleyDB resampled to 16000 Hz
    "context_width": 466,
    "note_range": 128,
    "samplerate": 16000,
    "input_normalization": False
}

name = common.name(args, "crepe_8mult"+("_normalized" if args["input_normalization"] else ""))

# To restore model from a checkpoint
# args["logdir"] = "..."

# Construct the network
network = NetworkMelody(threads=args["threads"])
print()
print(name)
print()
with network.session.graph.as_default():
    sess = network.session

    def preload_fn(aa): return aa.audio.load_resampled_audio(args["samplerate"])

    def dataset_transform(dataset):
        return dataset.batch(128).prefetch(1)

    def dataset_transform_train(dataset):
        return dataset.shuffle(20000).batch(args["batch_size"]).prefetch(1)

    mdb_stem_synth_train, mdb_stem_synth_validation, mdb_stem_synth_small_validation = common.prepare_mdb_stem_synth(preload_fn)

    train_dataset = datasets.AADataset(mdb_stem_synth_train, args, dataset_transform_train)
    mdb_stem_synth_validation_dataset = datasets.AADataset(mdb_stem_synth_validation, args, dataset_transform)
    mdb_stem_synth_small_validation_dataset = datasets.AADataset(mdb_stem_synth_small_validation, args, dataset_transform)

    network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes, dataset_preload_fn=preload_fn, dataset_transform=dataset_transform)

epochs = 10

validation_datasets = [
    VD(datasets.mdb_stem_synth.prefix+"_small", mdb_stem_synth_small_validation_dataset, 1000, True),
    VD(datasets.mdb_stem_synth.prefix, mdb_stem_synth_validation_dataset, 10000, False),
]

network.train(train_dataset, epochs, validation_datasets, save_every_n_batches=10000)
network.save()

print("ORCHSET evaluation")
valid_data_orchset = datasets.orchset.dataset("data/Orchset/")
orchset_dataset = datasets.AADataset(valid_data_orchset, args, dataset_transform)
network.evaluate(orchset_dataset, print_detailed=True)
