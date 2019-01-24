import os

from .common import melody_dataset_generator, load_melody_dataset

prefix = "mdb_stem_synth"


def generator(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "audio_stems")
    dataset_annot_path = os.path.join(dataset_root, "annotation_stems")

    return melody_dataset_generator(dataset_audio_path, dataset_annot_path, audio_suffix=".RESYN.wav", annot_suffix=".RESYN.csv")


def dataset(dataset_root):
    return load_melody_dataset(prefix, generator(dataset_root))
