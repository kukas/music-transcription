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


def prepare(preload_fn):
    wjazzd_split = get_split()

    def wjazzd_gen_split(name):
        gen = generator(os.path.join(modulepath, "..", "data", "WJazzD"))
        return filter(lambda x: x.uid in wjazzd_split[name], gen)

    train_data = load_melody_dataset(prefix, wjazzd_gen_split("train"))
    valid_data = load_melody_dataset(prefix, wjazzd_gen_split("validation"))

    for aa in train_data+valid_data:
        preload_fn(aa)

    # TODO: choose better small validation
    small_validation_data = [
        valid_data[14].slice(10, 20),
        valid_data[49].slice(10, 20),
    ]

    return train_data, valid_data, small_validation_data
