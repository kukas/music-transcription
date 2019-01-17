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
        notes = safe_hz_to_midi(freqs)
        notes = np.expand_dims(notes, axis=1)

        annotation = Annotation(times, notes)

        annotated_audios.append(AnnotatedAudio(annotation, audio))
        print(".", end=("" if (i+1) % 20 else "\n"))
    print()
    
    return annotated_audios

def safe_hz_to_midi(freqs):
    freqs = np.array(freqs)
    zeros = freqs==0
    freqs[zeros] = 1
    notes = mir_eval.util.hz_to_midi(freqs)

    notes[zeros] = 0
    return notes