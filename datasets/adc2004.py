import os
from .common import melody_dataset_generator, load_melody_dataset

modulepath = os.path.dirname(os.path.abspath(__file__))

prefix = "adc04"


def generator(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "adc2004_full_set")
    dataset_annot_path = os.path.join(dataset_root, "adc2004_full_set")

    return melody_dataset_generator(dataset_audio_path, dataset_annot_path, annot_suffix="REF.txt")


def dataset(dataset_root):
    return load_melody_dataset(prefix, generator(dataset_root))


def prepare(preload_fn):
    test_data = dataset(os.path.join(modulepath, "..", "data", "adc2004"))

    for aa in test_data:
        preload_fn(aa)

    return test_data
