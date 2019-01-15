import os
import mir_eval

from .common import safe_hz_to_midi
from .dataset import AnnotatedAudio, Audio, Annotation, Dataset, AADataset

def load_medleydb_melody_dataset(name, dataset_audio_path, dataset_annot_path, annot_extension=".csv"):
    uids = [f[:-4] for f in os.listdir(dataset_annot_path) if f.endswith(annot_extension)]

    annotated_audios = []
    for i, uid in enumerate(uids):
        # prepare audio
        audiopath = os.path.join(dataset_audio_path, uid, "{}.wav".format(uid))
        audio = Audio(audiopath, name+"_"+uid)

        # prepare annotation
        annotpath = os.path.join(dataset_annot_path, uid+annot_extension)
        times, freqs = mir_eval.io.load_time_series(annotpath, delimiter=",")
        notes = safe_hz_to_midi(freqs)
        annotation = Annotation(times, notes)

        annotated_audios.append(AnnotatedAudio(annotation, audio))
        print(".", end=("" if (i+1) % 20 else "\n"))
    print()
    
    return annotated_audios

def dataset(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "Audio")
    dataset_annot_path = os.path.join(dataset_root, "Annotations", "Melody_Annotations", "MELODY2")

    return load_medleydb_melody_dataset("MDB", dataset_audio_path, dataset_annot_path, annot_extension=".csv")