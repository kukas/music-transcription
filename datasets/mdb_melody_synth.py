import os
import mir_eval
from glob import glob

from .common import safe_hz_to_midi
from .dataset import AnnotatedAudio, Audio, Annotation, Dataset, AADataset

def load_mdb_melody_synth_melody_dataset(name, dataset_audio_path, dataset_annot_path, annot_extension=".csv"):
    uids = [f[:-17] for f in os.listdir(dataset_audio_path) if f.endswith(".wav")]

    annotated_audios = []
    for i, uid in enumerate(uids):
        # prepare audio
        audiopath = os.path.join(dataset_audio_path, "{}_MIX_melsynth.wav".format(uid))
        audio = Audio(audiopath, name+"_"+uid)

        # prepare annotation
        annotpath = glob(os.path.join(dataset_annot_path, uid+"_STEM*"))
        annotpath = annotpath[0]
        times, freqs = mir_eval.io.load_time_series(annotpath, delimiter=",")
        notes = safe_hz_to_midi(freqs)
        notes = np.expand_dims(notes, axis=1)
        annotation = Annotation(times, notes)

        annotated_audios.append(AnnotatedAudio(annotation, audio))
        print(".", end=("" if (i+1) % 20 else "\n"))
    print()
    
    return annotated_audios

def dataset(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "audio_mix")
    dataset_annot_path = os.path.join(dataset_root, "annotation_melody")

    return load_mdb_melody_synth_melody_dataset("MDB_melody_synth", dataset_audio_path, dataset_annot_path, annot_extension=".csv")