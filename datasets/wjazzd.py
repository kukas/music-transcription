import os
from .common import melody_dataset_generator, load_melody_dataset

prefix = "wjazzd"


def generator(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "audio")
    dataset_annot_path = os.path.join(dataset_root, "f0")

    return melody_dataset_generator(dataset_audio_path, dataset_annot_path, audio_suffix="_Solo.wav", annot_suffix="_FINAL_f0.csv")


def dataset(dataset_root):
    return load_melody_dataset(prefix, generator(dataset_root))
