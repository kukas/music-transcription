import os
from .common import load_mirex_melody_dataset

def dataset(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "audio", "mono")
    dataset_annot_path = os.path.join(dataset_root, "GT")

    return load_mirex_melody_dataset("orchset", dataset_audio_path, dataset_annot_path, annot_extension=".mel")