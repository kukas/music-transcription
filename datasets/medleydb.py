import os
import mir_eval
import numpy as np

from .common import melody_to_multif0
from .dataset import AnnotatedAudio, Audio, Annotation, Dataset, AADataset

def load_medleydb_melody_dataset(name, dataset_audio_path, dataset_annot_path, annot_extension=".csv"):
    uids = [f[:-len(annot_extension)] for f in os.listdir(dataset_annot_path) if f.endswith(annot_extension)]

    annotated_audios = []
    for i, uid in enumerate(uids):
        # prepare audio
        audiopath = os.path.join(dataset_audio_path, uid, "{}_MIX.wav".format(uid))
        audio = Audio(audiopath, name+"_"+uid)

        # prepare annotation
        annotpath = os.path.join(dataset_annot_path, uid+annot_extension)
        times, freqs = mir_eval.io.load_time_series(annotpath, delimiter=",")
        
        # minioptimalizace, ale doopravdy by se to mělo udělat líp
        notes = mir_eval.util.hz_to_midi(freqs)
        notes[freqs==0] = 0

        annotation = Annotation(times, melody_to_multif0(freqs), melody_to_multif0(notes))

        annotated_audios.append(AnnotatedAudio(annotation, audio))
        print(".", end=("" if (i+1) % 20 else "\n"))
    print()
    
    return annotated_audios

def dataset(dataset_root, split=None):
    dataset_audio_path = os.path.join(dataset_root, "Audio")
    dataset_annot_path = os.path.join(dataset_root, "Annotations", "Melody_Annotations", "MELODY2")
    mdb = "MDB"
    all_annotated_audios = load_medleydb_melody_dataset(mdb, dataset_audio_path, dataset_annot_path, annot_extension="_MELODY2.csv")
    if split:
        split_data = {}
        # prepare arrays
        for split_name in split.keys():
            split_data[split_name] = []

        # split the data
        for aa in all_annotated_audios:
            for split_name, split_uids in split.items():
                track_name = aa.audio.uid[len(mdb)+1:]
                if track_name in split_uids:
                    split_data[split_name].append(aa)
                    break
        return split_data
    else:
        return all_annotated_audios