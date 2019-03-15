import os
from glob import glob

from .common import melody_dataset_generator, load_melody_dataset, parallel_preload
from .medleydb import get_split

prefix = "mdb_mf0_synth"


def generator(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "audio_mix")
    dataset_annot_path = os.path.join(dataset_root, "annotation_mf0")

    return melody_dataset_generator(dataset_audio_path, dataset_annot_path, audio_suffix="_MIX_mf0synth.wav", annot_suffix="_MIX_mf0synth.csv")


def dataset(dataset_root):
    return load_melody_dataset(prefix, generator(dataset_root))


def prepare(preload_fn, threads=None):
    medleydb_split = get_split()

    def mdb_split(name):
        gen = generator("data/MDB-mf0-synth/")
        return filter(lambda x: x.uid in medleydb_split[name], gen)

    train_data = load_melody_dataset(prefix, mdb_split("train"))
    valid_data = load_melody_dataset(prefix, mdb_split("validation"))

    parallel_preload(preload_fn, train_data+valid_data, threads=threads)

    # TODO: choose better small validation
    small_validation_data = [
        valid_data[0].slice(250, 260),  # ženský zpěv s basovou kytarou, bubny a nějakými melodickými vysokými efekty
        valid_data[1].slice(60, 75),  # ženský zpěv s basou a bubny + chvíli sólo zpěv
        valid_data[3].slice(8, 18),  # basová kytara + divný šumivý zvuk
    ]

    return train_data, valid_data, small_validation_data
