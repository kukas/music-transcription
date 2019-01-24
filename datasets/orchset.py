import os
from .common import melody_dataset_generator, load_melody_dataset

prefix = "orchset"


def generator(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "audio", "mono")
    dataset_annot_path = os.path.join(dataset_root, "GT")

    return melody_dataset_generator(dataset_audio_path, dataset_annot_path, annot_suffix=".mel")


def dataset(dataset_root):
    return load_melody_dataset(prefix, generator(dataset_root))
