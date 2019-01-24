
import os
from .common import melody_dataset_generator, load_melody_dataset

prefix = "mirex05"


def generator(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "mirex05TrainFiles")
    dataset_annot_path = os.path.join(dataset_root, "mirex05TrainFiles")

    return melody_dataset_generator(dataset_audio_path, dataset_annot_path, annot_suffix="REF.txt")


def dataset(dataset_root):
    return load_melody_dataset(prefix, generator(dataset_root))
