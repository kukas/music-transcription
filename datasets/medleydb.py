import os
import mir_eval
import numpy as np
import json

from .common import melody_dataset_generator, load_melody_dataset
modulepath = os.path.dirname(os.path.abspath(__file__))

prefix = "mdb"

def get_split():
    # For MDB and MDB synth we use the train/validation/test split according to deepsalience paper
    with open(os.path.join(modulepath, "..", "data", "MedleyDB", "dataset_ismir_split.json")) as f:
        split = json.load(f)
    return split


def generator(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "Audio", "*")
    dataset_annot_path = os.path.join(dataset_root, "Annotations", "Melody_Annotations", "MELODY2")

    return melody_dataset_generator(dataset_audio_path, dataset_annot_path, audio_suffix="_MIX.wav", annot_suffix="_MELODY2.csv")


def dataset(dataset_root):
    return load_melody_dataset(prefix, generator(dataset_root))


def prepare(preload_fn):
    medleydb_split = get_split()

    def mdb_split(name):
        gen = generator(os.path.join(modulepath, "..", "data", "MedleyDB", "MedleyDB"))
        return filter(lambda x: x.uid in medleydb_split[name], gen)

    train_data = load_melody_dataset(prefix, mdb_split("train"))
    test_data = load_melody_dataset(prefix, mdb_split("test"))
    valid_data = load_melody_dataset(prefix, mdb_split("validation"))

    for aa in train_data+test_data+valid_data:
        preload_fn(aa)

    # TODO: choose better small validation
    small_validation_data = [
        valid_data[3].slice(15, 20.8),
        valid_data[9].slice(56, 61.4),
        valid_data[5].slice(55.6, 61.6),
    ]

    return train_data, test_data, valid_data, small_validation_data
