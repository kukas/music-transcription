from .dataset import AnnotatedAudio, Audio, Annotation, AADataset

import numpy as np
import mir_eval
import os

from collections import namedtuple

Track = namedtuple("Track", ("audio_path", "annot_path", "uid"))

def melody_dataset_generator(dataset_audio_path, dataset_annot_path, audio_suffix=".wav", annot_suffix=".csv"):
    uids = [f[:-len(audio_suffix)] for f in os.listdir(dataset_audio_path) if f.endswith(audio_suffix)]

    for uid in uids:
        audio_path = os.path.join(dataset_audio_path, uid+audio_suffix)
        annot_path = os.path.join(dataset_annot_path, uid+annot_suffix)
        
        if os.path.isfile(annot_path):
                yield Track(audio_path, annot_path, uid)


def load_melody_dataset(name, dataset_iterator):
    annotated_audios = []
    for i, (audio_path, annot_path, uid) in enumerate(dataset_iterator):
        # prepare audio
        audio = Audio(audio_path, name+"_"+uid)

        # prepare annotation
        annotation = None
        if annot_path is not None:
                times, freqs = mir_eval.io.load_time_series(annot_path, delimiter=r'\s+|,')
                freqs = np.expand_dims(freqs, 1)
                annotation = Annotation(times, freqs)

        annotated_audios.append(AnnotatedAudio(annotation, audio))

        print(".", end=("" if (i+1) % 20 else "\n"))
    print()

    assert len(annotated_audios) > 0

    return annotated_audios


def melody_to_multif0(values):
    return [[x] if x > 0 else [] for x in values]

def multif0_to_melody(values):
    return np.array([x[0] if len(x) > 0 else 0 for x in values])

def _hz_to_midi_safe(x):
    if x > 0:
        return mir_eval.util.hz_to_midi(x)
    else:
        return 0

hz_to_midi_safe = np.vectorize(_hz_to_midi_safe, otypes=[float])

def _midi_to_hz_safe(x):
    if x > 0:
        return mir_eval.util.midi_to_hz(x)
    else:
        return 0


midi_to_hz_safe = np.vectorize(_midi_to_hz_safe, otypes=[float])
