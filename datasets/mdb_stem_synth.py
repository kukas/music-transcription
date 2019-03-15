import os

from .common import melody_dataset_generator, load_melody_dataset, parallel_preload
from .medleydb import get_split

prefix = "mdb_stem_synth"


def generator(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "audio_stems")
    dataset_annot_path = os.path.join(dataset_root, "annotation_stems")

    return melody_dataset_generator(dataset_audio_path, dataset_annot_path, audio_suffix=".RESYN.wav", annot_suffix=".RESYN.csv")


def dataset(dataset_root):
    return load_melody_dataset(prefix, generator(dataset_root))


def prepare(preload_fn, threads=None):
    medleydb_split = get_split()

    def mdb_split(name):
        gen = generator("data/MDB-stem-synth/")
        return filter(lambda x: x.uid[:-len("_STEM_xx")] in medleydb_split[name], gen)

    train_data = load_melody_dataset(prefix, mdb_split("train"))
    valid_data = load_melody_dataset(prefix, mdb_split("validation"))

    parallel_preload(preload_fn, train_data+valid_data, threads)

    small_validation_data = [
        valid_data[3].slice(30, 40),  # nějaká kytara
        valid_data[4].slice(38, 50),  # zpěv ženský
        valid_data[5].slice(55, 65),  # zpěv mužský
        valid_data[13].slice(130, 140),  # zpěv mužský
    ]

    return train_data, valid_data, small_validation_data
