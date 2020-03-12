import os
import numpy as np
import mir_eval
import datasets
from .audio import check_dir
import csv
from madmom.io import midi
from intervaltree import IntervalTree

modulepath = os.path.dirname(os.path.abspath(__file__))
CACHED_FILES_PATH = os.path.join(modulepath, "..", "cached")

''' Handles the common time-frequency annotation format. '''
class Annotation:
    def __init__(self, times, freqs=None, notes=None, voicing=None):
        assert not (freqs is None and notes is None)

        self.times = np.array(times, dtype=np.float64)
        self.freqs = freqs
        self._freqs_mf0 = None
        self.notes = notes
        self._notes_mf0 = None
        self.voicing = voicing
        self._ref_matrix = None

        if freqs is None:
            self.freqs = datasets.common.midi_to_hz_safe(self.notes)

        if notes is None:
            self.notes = datasets.common.hz_to_midi_safe(self.freqs)

        if voicing is None:
            self.voicing = self.freqs[:, 0] > 0

        self.freqs = self.freqs.astype(np.float32)
        self.notes = self.notes.astype(np.float32)

    @staticmethod
    def from_time_series(annot_path, uid, hop_samples=256, unique_mf0=False):
        if annot_path is None:
            return None

        check_dir(CACHED_FILES_PATH)
        # Check if there is a cached numpy binary
        cached_path = os.path.join(CACHED_FILES_PATH, "{}_{}.npz".format(uid, hop_samples))
        if os.path.isfile(cached_path):
            times, freqs, notes, voicing = np.load(cached_path).values()
            return Annotation(times, freqs, notes, voicing)
        else:
            delimiter = r'\s+|,'
            times, freqs = mir_eval.io.load_ragged_time_series(annot_path, delimiter=delimiter)
            max_polyphony = np.max([len(frame) for frame in freqs])

            # resample annotations
            times_new = np.arange(times[0], times[-1], hop_samples/44100)
            if max_polyphony == 1:
                times, freqs = mir_eval.io.load_time_series(annot_path, delimiter=delimiter)
                voicing = freqs > 0
                freqs, voicing = mir_eval.melody.resample_melody_series(times, freqs, voicing, times_new, kind='linear')

                freqs_aligned = np.expand_dims(freqs, -1)
            else:
                freqs = mir_eval.multipitch.resample_multipitch(times, freqs, times_new)
                freqs_aligned = np.zeros((len(freqs), max_polyphony))
                voicing = np.zeros((len(freqs),), dtype=np.int32)
                for i, frame in enumerate(freqs):
                    for j, freq in enumerate(frame):
                        if freq == 0:
                            break
                        freqs_aligned[i, j] = freq
                        voicing[i] += 1

            annot = Annotation(times_new, freqs_aligned, voicing=voicing)
            if unique_mf0:
                annot = annot.unique_mf0()

            np.savez(cached_path, annot.times, annot.freqs, annot.notes, annot.voicing)

            return annot

    @staticmethod
    def from_midi(annot_path, uid, hop_samples=256, unique_mf0=True):
        check_dir(CACHED_FILES_PATH)
        # Check if there is a cached numpy binary
        cached_path = os.path.join(CACHED_FILES_PATH, "{}_{}.npz".format(uid, hop_samples))
        if os.path.isfile(cached_path):
            times, freqs, notes, voicing = np.load(cached_path).values()
            return Annotation(times, freqs, notes, voicing)
        else:
            pattern = midi.MIDIFile(annot_path)
            tree = IntervalTree()

            for onset, _pitch, duration, velocity, _channel in pattern.notes:
                pitch = int(_pitch)
                frame_start = onset
                frame_end = onset + duration
                tree[frame_start:frame_end] = pitch
            
            max_time = max(tree)[1]
            times = np.arange(0, max_time, hop_samples/44100)
            notes_mf0 = []
            for t in times:
                notes_at_t = []
                for note in tree[t]:
                    notes_at_t.append(note.data)
                if unique_mf0:
                    # remove duplicate notes
                    notes_at_t = list(set(notes_at_t))
                notes_mf0.append(notes_at_t)
            max_polyphony = np.max([len(frame) for frame in notes_mf0])

            freqs = np.zeros((len(times), max_polyphony))
            notes = np.zeros((len(times), max_polyphony))
            voicing = np.zeros((len(times),), dtype=np.int32)

            for i, notes_at_i in enumerate(notes_mf0):
                for j, note in enumerate(notes_at_i):
                    notes[i, j] = note
                    freqs[i, j] = mir_eval.util.midi_to_hz(note)
                    voicing[i] += 1

            annot = Annotation(times, freqs, notes, voicing)

            np.savez(cached_path, annot.times, annot.freqs, annot.notes, annot.voicing)

            return annot

    @staticmethod
    def from_musicnet_csv(annot_path, uid, hop_samples=256, unique_mf0=True, annot="instrument", instrument_mappings=None):
        check_dir(CACHED_FILES_PATH)
        # Check if there is a cached numpy binary
        cached_path = os.path.join(CACHED_FILES_PATH, "{}_{}.npz".format(uid, hop_samples))
        if os.path.isfile(cached_path):
            times, freqs, notes, voicing = np.load(cached_path).values()
            return Annotation(times, freqs, notes, voicing)
        else:
            print("generating from", annot_path)
            tree = IntervalTree()

            with open(annot_path, 'r') as f:
                reader = csv.DictReader(f, delimiter=',')
                for label in reader:
                    start_time = int(label['start_time'])/44100
                    end_time = int(label['end_time'])/44100

                    # instrument = int(label['instrument'])
                    # note = int(label['note'])
                    # start_beat = float(label['start_beat'])
                    # end_beat = float(label['end_beat'])
                    # note_value = label['note_value']
                    if annot == "instrument":
                        instrument_midi = int(label['instrument'])
                        instrument_id = instrument_mappings[instrument_midi]["id"]
                        
                        tree[start_time:end_time] = instrument_id

            max_time = max(tree)[1]
            times = np.arange(0, max_time, hop_samples/44100)
            notes_mf0 = []
            for t in times:
                notes_at_t = []
                for note in tree[t]:
                    notes_at_t.append(note.data)
                if unique_mf0:
                    # remove duplicate notes
                    notes_at_t = list(set(notes_at_t))
                notes_mf0.append(notes_at_t)
            max_polyphony = np.max([len(frame) for frame in notes_mf0])

            # freqs = np.zeros((len(times), max_polyphony))
            notes = np.full((len(times), max_polyphony), -1)
            voicing = np.zeros((len(times),), dtype=np.int32)

            for i, notes_at_i in enumerate(notes_mf0):
                for j, note in enumerate(notes_at_i):
                    notes[i, j] = note
                    # freqs[i, j] = mir_eval.util.midi_to_hz(note)
                    voicing[i] += 1

            annot = Annotation(times, np.array([]), notes, voicing)

            np.savez(cached_path, annot.times, annot.freqs, annot.notes, annot.voicing)

            return annot

    @property
    def max_polyphony(self):
        return np.max([len(notes_frame) for notes_frame in self.notes])

    @property
    def notes_mf0(self):
        if self._notes_mf0 is None:
            self._notes_mf0 = [np.array(frame[:v]) for frame, v in zip(self.notes, self.voicing)]
        return self._notes_mf0

    @property
    def freqs_mf0(self):
        if self._freqs_mf0 is None:
            self._freqs_mf0 = [np.array(frame[frame > 0]) for frame in self.freqs]
        return self._freqs_mf0
    
    def ref_matrix(self, note_range, min_note):
        size = note_range - min_note
        if self._ref_matrix is None or self._ref_matrix.shape[1] != note_range:
            self._ref_matrix = np.full((len(self.times), size), False)
            for t, notes_frame in enumerate(self.notes_mf0):
                for _note in notes_frame:
                    # _note has type float32
                    note = int(_note) - min_note
                    self._ref_matrix[t, note] = True

        return self._ref_matrix

    def get_frame_width(self):
        if len(self.times) == 0:
            raise RuntimeError("The annotation is empty.")
        return self.times[1]-self.times[0]

    def slice(self, start, end):
        framerate = 1/self.get_frame_width()
        b0, b1 = int(start*framerate), int(end*framerate)
        sliced_times = self.times[b0:b1]
        # time offset
        sliced_times = sliced_times - sliced_times[0]
        sliced_freqs = self.freqs[b0:b1]
        sliced_notes = self.notes[b0:b1]
        sliced_voicing = self.voicing[b0:b1]
        return Annotation(sliced_times, sliced_freqs, sliced_notes, sliced_voicing)

    def unique_mf0(self):
        self._freqs_mf0 = None
        self._notes_mf0 = None

        for i in range(len(self.freqs)):
            for j in range(len(self.freqs[i])):
                if self.freqs[i, j] == 0.0:
                    continue
                for k in range(j+1, len(self.freqs[i])):
                    if np.isclose(self.freqs[i, j], self.freqs[i, k]):
                        self.freqs[i, k] = 0.0
                        self.notes[i, k] = 0.0
        return self
