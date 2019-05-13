import os

from .common import melody_dataset_generator, load_melody_dataset, parallel_preload
from .medleydb import get_split

modulepath = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(modulepath, "..", "data")

prefix = "mdb_stem_synth"


def generator():
    dataset_audio_path = os.path.join(data_root, "MDB-stem-synth", "audio_stems")
    dataset_annot_path = os.path.join(data_root, "MDB-stem-synth", "annotation_stems")

    return melody_dataset_generator(dataset_audio_path, dataset_annot_path, audio_suffix=".RESYN.wav", annot_suffix=".RESYN.csv")


def dataset():
    return load_melody_dataset(prefix, generator())


def prepare(preload_fn, threads=None):
    medleydb_split = get_split()

    def mdb_split(name):
        gen = generator()
        return filter(lambda x: x.track_id[:-len("_STEM_xx")] in medleydb_split[name], gen)

    train_data = load_melody_dataset(prefix, mdb_split("train"))
    test_data = load_melody_dataset(prefix, mdb_split("test"))
    valid_data = load_melody_dataset(prefix, mdb_split("validation"))

    parallel_preload(preload_fn, train_data+test_data+valid_data, threads)

    small_validation_data = [
        valid_data[10].slice(0, 20),  # basová kytara
        valid_data[12].slice(125, 135),  # basová vybrnkávání
        valid_data[4].slice(40, 50),  # kytara
        valid_data[0].slice(5, 10),  # zpěv mužský
        valid_data[15].slice(60, 70),  # zpěv ženský
    ]

    return train_data, test_data, valid_data, small_validation_data
