import os
from .common import load_mirex_melody_dataset

def dataset(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "mirex05TrainFiles")
    dataset_annot_path = os.path.join(dataset_root, "mirex05TrainFiles")

    return load_mirex_melody_dataset("mirex05", dataset_audio_path, dataset_annot_path, annot_extension="REF.txt")