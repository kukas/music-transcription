import os
from glob import glob

from .common import melody_dataset_generator, load_melody_dataset
from .medleydb import get_split

prefix = "mdb_melody_synth"


def generator(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "audio_mix")
    dataset_annot_path = os.path.join(dataset_root, "annotation_melody")

    return melody_dataset_generator(dataset_audio_path, dataset_annot_path, audio_suffix="_MIX_melsynth.wav", annot_suffix="_STEM*.csv")


def dataset(dataset_root):
    return load_melody_dataset(prefix, generator(dataset_root))


def prepare(preload_fn):
    medleydb_split = get_split()

    def mdb_split(name):
        gen = generator("data/MDB-melody-synth/")
        return filter(lambda x: x.uid in medleydb_split[name], gen)

    train_data = load_melody_dataset(prefix, mdb_split("train"))
    valid_data = load_melody_dataset(prefix, mdb_split("validation"))

    for aa in train_data+valid_data:
        preload_fn(aa)

    # TODO: choose better small validation
    small_validation_data = [
        valid_data[0].slice(15, 25), # ženský zpěv
        valid_data[2].slice(0, 10), # mužský zpěv
        valid_data[6].slice(0,10), # kytara (souzvuk)
        valid_data[7].slice(12, 20),  # basová kytara+piano bez melodie, pak zpěvačka
    ]

    return train_data, valid_data, small_validation_data
