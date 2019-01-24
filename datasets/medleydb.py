import os
import mir_eval
import numpy as np

from .common import melody_dataset_generator, load_melody_dataset

prefix = "mdb"


def generator(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "Audio")
    dataset_annot_path = os.path.join(dataset_root, "Annotations", "Melody_Annotations", "MELODY2")

    return melody_dataset_generator(dataset_audio_path, dataset_annot_path, audio_suffix="_MIX.wav", annot_suffix="_MELODY2.csv")


def dataset(dataset_root):
    return load_melody_dataset(prefix, generator(dataset_root))
