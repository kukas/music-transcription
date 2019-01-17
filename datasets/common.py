from .dataset import AnnotatedAudio, Audio, Annotation, Dataset, AADataset

import numpy as np
import mir_eval
import os

# This could be rewritten using glob.glob to generalize for MedleyDB (all the Audio files are in their respective folder)
def load_mirex_melody_dataset(name, dataset_audio_path, dataset_annot_path, annot_extension=".csv"):
    uids = [f[:-4] for f in os.listdir(dataset_audio_path) if f.endswith('.wav')]

    annotated_audios = []
    for i, uid in enumerate(uids):
        # prepare audio
        audiopath = os.path.join(dataset_audio_path, "{}.wav".format(uid))
        audio = Audio(audiopath, name+"_"+uid)

        # prepare annotation
        annotpath = os.path.join(dataset_annot_path, uid+annot_extension)
        times, freqs = mir_eval.io.load_time_series(annotpath)

        annotation = Annotation(times, melody_to_multif0(freqs))

        annotated_audios.append(AnnotatedAudio(annotation, audio))
        print(".", end=("" if (i+1) % 20 else "\n"))
    print()
    
    return annotated_audios

def melody_to_multif0(values):
    return [[x] if x > 0 else [] for x in values]

def multif0_to_melody(values):
    return [x[0] if len(x) > 0 else 0 for x in values]