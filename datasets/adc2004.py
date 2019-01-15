import os
from .common import load_mirex_melody_dataset

def dataset(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "adc2004_full_set")
    dataset_annot_path = os.path.join(dataset_root, "adc2004_full_set")

    return load_mirex_melody_dataset("adc2004", dataset_audio_path, dataset_annot_path, annot_extension="REF.txt")