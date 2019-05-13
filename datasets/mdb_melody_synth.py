import os
from glob import glob

from .common import melody_dataset_generator, load_melody_dataset, parallel_preload
from .medleydb import get_split

modulepath = os.path.dirname(os.path.abspath(__file__))

prefix = "mdb_melody_synth"

def generator(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "audio_mix")
    dataset_annot_path = os.path.join(dataset_root, "annotation_melody")

    return melody_dataset_generator(dataset_audio_path, dataset_annot_path, audio_suffix="_MIX_melsynth.wav", annot_suffix="_STEM*.csv")


def dataset(dataset_root):
    return load_melody_dataset(prefix, generator(dataset_root))


def prepare(preload_fn, threads=None):
    medleydb_split = get_split()

    def mdb_split(name):
        gen = generator(os.path.join(modulepath, "..", "data", "MDB-melody-synth"))
        return filter(lambda x: x.track_id in medleydb_split[name], gen)

    train_data = load_melody_dataset(prefix, mdb_split("train"))
    test_data = load_melody_dataset(prefix, mdb_split("test"))
    valid_data = load_melody_dataset(prefix, mdb_split("validation"))

    parallel_preload(preload_fn, train_data+test_data+valid_data, threads=threads)

    # TODO: choose better small validation
    small_validation_data = [
        valid_data[0].slice(15, 25), # ženský zpěv
        valid_data[2].slice(0, 10), # mužský zpěv
        valid_data[6].slice(0,10), # kytara (souzvuk)
        valid_data[7].slice(12, 20),  # basová kytara+piano bez melodie, pak zpěvačka
    ]

    return train_data, test_data, valid_data, small_validation_data
