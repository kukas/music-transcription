
import os
from .common import melody_dataset_generator, load_melody_dataset, parallel_preload

modulepath = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(modulepath, "..", "data")

prefix = "mirex05"


def generator():
    dataset_audio_path = os.path.join(data_root, "mirex05TrainFiles")
    dataset_annot_path = os.path.join(data_root, "mirex05TrainFiles")

    return melody_dataset_generator(dataset_audio_path, dataset_annot_path, annot_suffix="REF.txt")


def dataset():
    return load_melody_dataset(prefix, generator())


def prepare(preload_fn, threads=None):
    test_data = dataset()

    parallel_preload(preload_fn, test_data, threads=threads)

    return test_data
