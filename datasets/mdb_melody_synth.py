import os
import mir_eval
from glob import glob

from .common import load_melody_dataset, Track

prefix = "mdb_melody_synth"

def generator(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "audio_mix")
    dataset_annot_path = os.path.join(dataset_root, "annotation_melody")
    annot_extension="_STEM*.csv"

    uids = [f[:-17] for f in os.listdir(dataset_audio_path) if f.endswith(".wav")]

    for uid in uids:
        audio_path = os.path.join(dataset_audio_path, "{}_MIX_melsynth.wav".format(uid))
        annot_path = glob(os.path.join(dataset_annot_path, uid+annot_extension))
        annot_path = annot_path[0]

        yield Track(audio_path, annot_path, uid)

def dataset(dataset_root):
    return load_melody_dataset(prefix, generator(dataset_root))
