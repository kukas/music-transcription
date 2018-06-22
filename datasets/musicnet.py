import csv
import os
import numpy as np

from intervaltree import IntervalTree

from .dataset import Audio, Annotation, AnnotatedAudio

def musicnet_datasets(musicnet_root, samplerate):
    def process_labels_file(path, max_time, hop=0.01):
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
        times = np.arange(0, max_time, hop)
        notes = [[midinote[2][1] for midinote in midinotes[t]] for t in times]
        return Annotation(times, notes)

    def uids(folder):
        return [int(item[:4]) for item in os.listdir(os.path.join(musicnet_root, folder)) if item.endswith('.csv')]

    def load_musicnet_annotaudios(split_name):
        annotated_audios = []
        _uids = uids(split_name+"_labels")
        for uid in _uids:
            # prepare audio
            audiopath = os.path.join(musicnet_root, split_name+"_data", "{}.wav".format(uid))
            audio = Audio(audiopath, "musicnet_{}_{}".format(split_name, uid))
            audio.load_resampled_audio(samplerate)
            max_time = audio.get_duration()

            # prepare annotation
            annotpath = os.path.join(musicnet_root, split_name+"_labels", "{}.csv".format(uid))
            annotation = process_labels_file(annotpath, max_time)
            annotated_audios.append(AnnotatedAudio(annotation, audio))

        return annotated_audios

    test = load_musicnet_annotaudios("test")
    # train = load_musicnet_annotaudios("train")

    # return test, train
    return test
