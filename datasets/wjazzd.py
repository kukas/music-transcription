import os
import json
from .common import melody_dataset_generator, load_melody_dataset

modulepath = os.path.dirname(os.path.abspath(__file__))

prefix = "wjazzd"

def get_split():
    with open(os.path.join(modulepath, "..", "data", "WJazzD", "wjazzd_split.json")) as f:
        split = json.load(f)
    return split


def generator(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "audio_f0")
    dataset_annot_path = os.path.join(dataset_root, "f0")

    return melody_dataset_generator(dataset_audio_path, dataset_annot_path, audio_suffix="_Solo.wav", annot_suffix="_FINAL_f0.csv")


def dataset(dataset_root):
    return load_melody_dataset(prefix, generator(dataset_root))
