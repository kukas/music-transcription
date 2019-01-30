import tensorflow as tf
import numpy as np

import datasets
from model import NetworkMelody, VD

import common

def create_model(self, args):
    annotations = self.annotations[:, :, 0]
    window = self.window[:, :-1]
    voicing_ref = tf.cast(tf.layers.flatten(tf.greater(annotations, 0)), tf.float32)

    if args["input_normalization"]:
        mean, var = tf.nn.moments(window, axes=[1])
        mean = tf.expand_dims(mean, axis=1)

        epsilon = 1e-3
        std_inv = tf.math.rsqrt(var + epsilon)
        std_inv = tf.expand_dims(std_inv, axis=1)

        window = (window - mean) * std_inv


    melody = tf.layers.dense(window, 1024, activation=tf.nn.tanh)
    output_layer = tf.layers.dense(melody, self.note_range, activation=None)

    voicing = tf.layers.dense(window, 1024, activation=tf.nn.tanh)
    voicing = tf.layers.dense(voicing, 1, activation=None)
    voicing_est = tf.cast(tf.greater(voicing, 0), tf.int32)

    self.note_logits = tf.reshape(output_layer, [-1, self.annotations_per_window, self.note_range])
    self.note_probabilities = tf.nn.softmax(self.note_logits)
    self.est_notes = tf.argmax(self.note_logits, axis=2, output_type=tf.int32)
    self.est_notes = self.est_notes * voicing_est

    self.loss = tf.losses.sparse_softmax_cross_entropy(annotations, self.note_logits, weights=voicing_ref)
    self.loss += tf.losses.sigmoid_cross_entropy(voicing_ref, voicing)

    self.training = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

args = {
    "threads": 6,
    "batch_size": 32,
    "annotations_per_window": 1,
    "frame_width": round(256/(44100/16000)), # frame_width of MedleyDB resampled to 16000 Hz
    "context_width": 466,
    "note_range": 128,
    "samplerate": 16000,
    "input_normalization": False
}

# To restore model from a checkpoint
# args["logdir"] = "..."

name_normalized = ("_normalized" if args["input_normalization"] else "")
common.name(args, "baseline_tanh_{}".format(name_normalized))


# Construct the network
network = NetworkMelody(threads=args["threads"])

with network.session.graph.as_default():

    def preload_fn(aa):
        return aa.audio.load_resampled_audio(args["samplerate"])

    def dataset_transform(dataset):
        return dataset.batch(128).prefetch(1)

    def dataset_transform_train(dataset):
        return dataset.shuffle(20000).batch(args["batch_size"]).prefetch(1)

    train, validation, small_validation = datasets.medleydb.prepare(preload_fn)

    train_dataset = datasets.AADataset(train, args, dataset_transform_train)
    validation_dataset = datasets.AADataset(validation, args, dataset_transform)
    small_validation_dataset = datasets.AADataset(small_validation, args, dataset_transform)

    _, _, mdb_stem_synth_small_validation = datasets.mdb_stem_synth.prepare(preload_fn)
    mdb_stem_synth_small_validation_dataset = datasets.AADataset(mdb_stem_synth_small_validation, args, dataset_transform)

    network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes, dataset_preload_fn=preload_fn, dataset_transform=dataset_transform)

epochs = 1
validation_datasets = [
    VD(datasets.mdb_stem_synth.prefix+"_small", mdb_stem_synth_small_validation_dataset, 1000, True),
    VD(datasets.mdb_melody_synth.prefix+"_small", small_validation_dataset, 1000, True),
    VD(datasets.mdb_melody_synth.prefix, validation_dataset, 10000, False),
]

network.train(train_dataset, epochs, validation_datasets, save_every_n_batches=10000)
network.save()

# print("ORCHSET evaluation")
# valid_data_orchset = datasets.orchset.dataset("data/Orchset/")
# orchset_dataset = datasets.AADataset(valid_data_orchset, args, dataset_transform)
# network.evaluate(orchset_dataset, print_detailed=True)
