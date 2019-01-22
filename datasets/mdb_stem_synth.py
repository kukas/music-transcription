import os
from glob import glob

from .common import load_melody_dataset, Track

prefix = "mdb_stem_synth"

def generator(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "audio_stems")
    dataset_annot_path = os.path.join(dataset_root, "annotation_stems")
    annot_extension = "_STEM*.csv"
    audio_extension = ".RESYN.wav"

    uids = [f[:-len(audio_extension)] for f in os.listdir(dataset_audio_path) if f.endswith(audio_extension)]

    for uid in uids:
        audio_path = os.path.join(dataset_audio_path, uid+audio_extension)
        annot_path = glob(os.path.join(dataset_annot_path, uid+annot_extension))
        annot_path = annot_path[0]

        yield Track(audio_path, annot_path, uid)

def dataset(dataset_root):
    return load_melody_dataset(prefix, generator(dataset_root))
