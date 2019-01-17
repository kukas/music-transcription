import csv
import os
import numpy as np
import pickle

from intervaltree import IntervalTree

from .dataset import Audio, Annotation, AnnotatedAudio

def process_labels_file(path, hop=0.01):
    preprocessed_path = path+".npz"
    if os.path.isfile(preprocessed_path):
        times, notes = pickle.load(open(preprocessed_path, "rb"))
        return Annotation(times, notes)

    midinotes = IntervalTree()
    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for label in reader:
            start_time = int(label['start_time'])/44100
            end_time = int(label['end_time'])/44100
            instrument = int(label['instrument'])
            note = int(label['note'])
            start_beat = float(label['start_beat'])
            end_beat = float(label['end_beat'])
            note_value = label['note_value']
            midinotes[start_time:end_time] = (instrument,note,start_beat,end_beat,note_value)

    max_time = max(midinotes)[1]
    times = np.arange(0, max_time, hop)
    notes = [[midinote[2][1] for midinote in midinotes[t]] for t in times]

    pickle.dump((times, notes), open(preprocessed_path,"wb"))

    return Annotation(times, notes=notes)

def musicnet_load_uids(musicnet_root, split_name, uids):
    annotated_audios = []
    for i, uid in enumerate(uids):
        # prepare audio
        audiopath = os.path.join(musicnet_root, split_name+"_data", "{}.wav".format(uid))
        audio = Audio(audiopath, "musicnet_{}_{}".format(split_name, uid))

        # prepare annotation
        annotpath = os.path.join(musicnet_root, split_name+"_labels", "{}.csv".format(uid))
        annotation = process_labels_file(annotpath)
        annotated_audios.append(AnnotatedAudio(annotation, audio))
        print(".", end=("" if (i+1) % 20 else "\n"))
    print()
    
    return annotated_audios

def musicnet_dataset(musicnet_root, split_name, first_n=0):
    def uids(folder):
        return [int(item[:4]) for item in os.listdir(os.path.join(musicnet_root, folder)) if item.endswith('.csv')]

    _uids = uids(split_name+"_labels")
    if first_n:
        _uids = _uids[:first_n]

    return musicnet_load_uids(musicnet_root, split_name, _uids)