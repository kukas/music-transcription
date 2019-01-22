import tensorflow as tf
import numpy as np

import datasets
from model import NetworkMelody, VD

import common

def create_model(self, args):
    annotations = self.annotations[:, :, 0]
    window = self.window[:, :-1]
    voicing_ref = tf.cast(tf.layers.flatten(tf.greater(annotations, 0)), tf.float32)

    # Simple one layer model
    dense = tf.layers.dense(window, 1024, activation=tf.nn.relu)
    output_layer = tf.layers.dense(dense, self.note_range, activation=None)

    self.note_probabilites = tf.reshape(output_layer, [-1, self.annotations_per_window, self.note_range])
    self.est_notes = tf.argmax(self.note_probabilites, axis=2, output_type=tf.int32)
    
    self.loss = tf.losses.sparse_softmax_cross_entropy(tf.layers.flatten(annotations), tf.layers.flatten(self.note_probabilites), weights=voicing_ref)

    self.training = tf.train.AdamOptimizer(0.0002).minimize(self.loss, global_step=self.global_step)

args = {
    "threads": 6,
    "batch_size": 32,
    "annotations_per_window": 1,
    "frame_width": round(256/(44100/16000)), # frame_width of MedleyDB resampled to 16000 Hz
    "context_width": 466,
    "note_range": 128,
    "samplerate": 16000
}

name = common.name(args, "baseline")
print(name)

# To restore model from a checkpoint
# args["logdir"] = "..."

# Construct the network
network = NetworkMelody(threads=args["threads"])

with network.session.graph.as_default():
    preload_hook = lambda aa: aa.audio.load_resampled_audio(args["samplerate"])
    dataset_transform = lambda dataset: dataset.batch(128).prefetch(1)
    dataset_transform_train = lambda dataset: dataset.shuffle(5000).batch(args["batch_size"]).prefetch(1)

    # # Prepare the data (and load annotations)
    # mdb_stem_synth_train, mdb_stem_synth_validation, mdb_stem_synth_small_validation = common.prepare_mdb_stem_synth()
    # # Prepare the datasets
    # train_dataset = datasets.AADataset(mdb_stem_synth_train, args, preload_hook, dataset_transform_train)
    # mdb_stem_synth_validation_dataset = datasets.AADataset(mdb_stem_synth_validation, args, preload_hook, dataset_transform)
    # mdb_stem_synth_small_validation_dataset = datasets.AADataset(mdb_stem_synth_small_validation, args, preload_hook, dataset_transform)
    
    # Prepare the data (and load annotations)
    medleydb_train, medleydb_validation, medleydb_small_validation = common.prepare_medleydb()
    # Prepare the datasets
    train_dataset = datasets.AADataset(medleydb_train, args, preload_hook, dataset_transform_train)
    medleydb_validation_dataset = datasets.AADataset(medleydb_validation, args, preload_hook, dataset_transform)
    medleydb_small_validation_dataset = datasets.AADataset(medleydb_small_validation, args, preload_hook, dataset_transform)

    network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes, dataset_preload_hook=preload_hook, dataset_transform=dataset_transform)


validation_datasets=[
    # VD(datasets.mdb_stem_synth.prefix, mdb_stem_synth_validation_dataset, 10000, False),
    # VD(datasets.mdb_stem_synth.prefix+"_small", mdb_stem_synth_small_validation_dataset, 1000, True),
    VD(datasets.medleydb.prefix, medleydb_validation_dataset, 10000, False),
    VD(datasets.medleydb.prefix+"_small", medleydb_small_validation_dataset, 1000, True),
]

epochs = 10
network.train(train_dataset, epochs, validation_datasets, save_every_n_batches=10000)

network.save()

print("ORCHSET evaluation")
valid_data_orchset = datasets.orchset.dataset("data/Orchset/")
orchset_dataset = datasets.AADataset(valid_data_orchset, args, preload_hook, dataset_transform)
network.evaluate(orchset_dataset, print_detailed=True)
