import os
import mir_eval
import numpy as np

from .common import load_melody_dataset, Track

prefix = "mdb"

def generator(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "Audio")
    dataset_annot_path = os.path.join(dataset_root, "Annotations", "Melody_Annotations", "MELODY2")
    annot_extension = "_MELODY2.csv"

    uids = [f[:-len(annot_extension)] for f in os.listdir(dataset_annot_path) if f.endswith(annot_extension)]
    for uid in uids:
        audio_path = os.path.join(dataset_audio_path, uid, "{}_MIX.wav".format(uid))
        annot_path = os.path.join(dataset_annot_path, uid+annot_extension)

        yield Track(audio_path, annot_path, uid)

def dataset(dataset_root):
    return load_melody_dataset(prefix, generator(dataset_root))
