import soundfile as sf
import numpy as np
import os
from resampy import resample
import librosa
import mir_eval
import tensorflow as tf

import datasets

np.random.seed(42)

PROCESSED_FILES_PATH = "./processed"
CACHED_FILES_PATH = "./cached"

def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

''' It is useful to keep audio and its annotation together because we can apply
transformations such as concatenation of training examples. '''
class AnnotatedAudio:
    def __init__(self, annotation, audio):
        self.audio = audio
        self.annotation = annotation

    def slice(self, start, end):
        return AnnotatedAudio(self.annotation.slice(start, end), self.audio.slice(start, end))

''' Audio container
Audio can be represented in different forms - as raw audio with various
sampling rates or as spectrograms. '''
class Audio:
    def __init__(self, path, uid):
        self.path = path
        self.filename = os.path.splitext(os.path.basename(path))[0]
        self.uid = uid
        self.samples = []
        self.samples_count = 0
        self.samplerate = 0
        self.spectrogram_hop_size = None
        self.spectrogram = None

    def load_resampled_audio(self, samplerate):
        assert self.samples == []

        check_dir(PROCESSED_FILES_PATH)
        resampled_path = os.path.join(PROCESSED_FILES_PATH, self.uid+"_{}.wav".format(samplerate))

        if not os.path.isfile(resampled_path):
            print("resampling", self.uid)
            audio, sr_orig = sf.read(self.path)
            print(audio.shape)
                    
            if len(audio.shape) >= 2: # mono downmixing, if needed
                audio = np.mean(audio, axis=1)
            audio_low = resample(audio, sr_orig, samplerate)

            sf.write(resampled_path, audio_low.astype(np.float32), samplerate)

        audio, sr_orig = sf.read(resampled_path, dtype="int16")
        self.samples = audio

        assert sr_orig == samplerate

        self.samples_count = len(self.samples)
        self.samplerate = samplerate
    
    def load_spectrogram(self, spec_function, spec_function_thumbprint, hop_size):
        self.spectrogram_hop_size = hop_size

        check_dir(PROCESSED_FILES_PATH)
        spec_path = os.path.join(PROCESSED_FILES_PATH, self.uid+"_{}.npy".format(spec_function_thumbprint))

        # spec_path = self.path.replace(".wav", spec_function_thumbprint+".npy")
        audio, samplerate = sf.read(self.path)

        if os.path.isfile(spec_path):
            spec = np.load(spec_path)
        else:
            spec = spec_function(audio, samplerate)
            np.save(spec_path, spec)
        
        self.spectrogram = spec
        self.samples_count = len(audio)
        self.samplerate = samplerate

    def get_duration(self):
        if self.samplerate == 0:
            return 0
        return self.samples_count/self.samplerate

    def get_padded_audio(self, start_sample, end_sample):
        if len(self.samples) == 0:
            return None

        cut_start = start_sample
        cut_end = end_sample

        last_sample_index = self.samples_count - 1

        if start_sample > last_sample_index:
            return np.zeros(end_sample-start_sample, dtype=np.int16)

        # padding the snippets with zeros when the context reaches outside the audio
        cut_start_diff = 0
        cut_end_diff = 0
        if start_sample < 0:
            cut_start = 0
            cut_start_diff = cut_start - start_sample
        if end_sample > last_sample_index:
            cut_end = last_sample_index
            cut_end_diff = end_sample - last_sample_index

        audio = self.samples[cut_start:cut_end]

        if cut_start_diff:
            zeros = np.zeros(cut_start_diff, dtype=np.int16)
            audio = np.concatenate([zeros, audio])

        if cut_end_diff:
            zeros = np.zeros(cut_end_diff, dtype=np.int16)
            audio = np.concatenate([audio, zeros])
        return audio


    def get_padded_spectrogram(self, start_sample, end_sample):
        if self.spectrogram is None:
            return None

        cut_length = int((end_sample - start_sample)/self.spectrogram_hop_size)

        cut_start = start_window = int(start_sample/self.spectrogram_hop_size)
        cut_end = end_window = cut_start + cut_length

        last_window_index = self.spectrogram.shape[1] - 1

        # padding the snippets with zeros when the context reaches outside the audio
        cut_start_diff = 0
        cut_end_diff = 0
        if start_window < 0:
            cut_start = 0
            cut_start_diff = cut_start - start_window
        if end_window > last_window_index:
            cut_end = last_window_index
            cut_end_diff = end_window - last_window_index

        spectrogram = self.spectrogram[:,cut_start:cut_end]

        if cut_start_diff:
            zeros = np.zeros(self.spectrogram.shape[:1]+(cut_start_diff,), dtype=self.spectrogram.dtype)
            spectrogram = np.concatenate([zeros, spectrogram], axis=1)

        if cut_end_diff:
            zeros = np.zeros(self.spectrogram.shape[:1]+(cut_end_diff,), dtype=self.spectrogram.dtype)
            spectrogram = np.concatenate([spectrogram, zeros], axis=1)
        
        return spectrogram

    def slice(self, start, end):
        if len(self.samples) == 0:
            raise Exception("Audio samples are empty")

        sliced = Audio(self.path, self.uid)
        sliced.samplerate = self.samplerate
        b0, b1 = int(start*self.samplerate), int(end*self.samplerate)
        sliced.samples = self.samples[b0:b1]

        if len(sliced.samples) == 0:
            raise Exception("Sliced audio samples are empty")

        sliced.samples_count = b1-b0

        if self.spectrogram is not None:
            sliced.spectrogram_hop_size = self.spectrogram_hop_size
            sliced.spectrogram = self.spectrogram[:,int(b0/self.spectrogram_hop_size):int(b1/self.spectrogram_hop_size)]

        return sliced
    
    def get_window_at_sample(self, window_start_sample, inner_window_size, context_width):
        window_end_sample = window_start_sample + inner_window_size
        
        b0 = int(window_start_sample-context_width)
        b1 = int(window_end_sample+context_width)

        return self.get_padded_audio(b0, b1), self.get_padded_spectrogram(b0, b1)

    def iterator(self, inner_window_size, context_width):
        for sample_index in range(0, self.samples_count, inner_window_size):
            yield self.get_window_at_sample(sample_index, inner_window_size, context_width)

''' Handles the common time-frequency annotation format. '''
class Annotation:
    def __init__(self, times, freqs=None, notes=None, voicing=None):
        assert not (freqs is None and notes is None)

        self.times = np.array(times, dtype=np.float32)
        self.freqs = freqs
        self._freqs_mf0 = None
        self.notes = notes
        self._notes_mf0 = None
        self.voicing = voicing

        if freqs is None:
            self.freqs = datasets.common.midi_to_hz_safe(self.notes)

        if notes is None:
            self.notes = datasets.common.hz_to_midi_safe(self.freqs)

        if voicing is None:
            self.voicing = self.freqs[:,0] > 0


        self.freqs = self.freqs.astype(np.float32)
        self.notes = self.notes.astype(np.float32)
    
    @staticmethod
    def from_time_series(annot_path, dataset_prefix):
        check_dir(CACHED_FILES_PATH)
        # Check if there is a cached numpy binary
        filename = os.path.splitext(os.path.basename(annot_path))[0]
        cached_path = os.path.join(CACHED_FILES_PATH, dataset_prefix+"_"+filename+".npz")
        if os.path.isfile(cached_path):
            times, freqs, notes, voicing = np.load(cached_path).values()
            return Annotation(times, freqs, notes, voicing)
        else:
            times, freqs = mir_eval.io.load_ragged_time_series(annot_path, delimiter=r'\s+|,')
            max_polyphony = np.max([len(frame) for frame in freqs])
            freqs_aligned = np.zeros((len(freqs), max_polyphony))
            voicing = np.zeros((len(freqs),), dtype=np.int32)
            for i, frame in enumerate(freqs):
                for j, freq in enumerate(frame):
                    if freq == 0:
                        break
                    freqs_aligned[i, j] = freq
                    voicing[i] += 1
            annot = Annotation(times, freqs_aligned, voicing=voicing)

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
            self._freqs_mf0 = [np.array(frame[:v]) for frame, v in zip(self.freqs, self.voicing)]
        return self._freqs_mf0

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

class AADataset:
    def __init__(self, _annotated_audios, args, dataset_transform=None, shuffle=False, hop_size=None):
        self._annotated_audios = _annotated_audios

        # self.frame_width = int(np.round(aa.annotation.get_frame_width()*self.samplerate))
        self.frame_width = args.frame_width

        self.context_width = args.context_width
        self.annotations_per_window = args.annotations_per_window
        self.hop_size = hop_size if hop_size is not None else self.annotations_per_window
        # todo: přejmenovat na window_width?
        self.inner_window_size = self.annotations_per_window*self.frame_width
        self.window_size = self.inner_window_size + 2*self.context_width

        for aa in self._annotated_audios:
            if aa.annotation is None:
                # add dummy annotation if the annotation is missing
                times = np.arange(0, aa.audio.samples_count, self.frame_width) / aa.audio.samplerate
                freqs = np.tile(440, [len(times),1])
                notes = np.tile(69, [len(times),1])

                aa.annotation = Annotation(times, freqs, notes)

        self.samplerate = _annotated_audios[0].audio.samplerate

        # generate example positions all at once so that we can shuffle them as a whole
        indices = []
        for aa_index, aa in enumerate(self._annotated_audios):
            if self.window_size > aa.audio.samples_count:
                raise RuntimeError("Window size is bigger than the audio.")

            annot_length = len(aa.annotation.times)
            annotation_indices = np.arange(0, annot_length, self.annotations_per_window, dtype=np.int32)
            aa_indices = np.full((len(annotation_indices),), aa_index, dtype=np.int32)

            indices.append(np.stack((aa_indices, annotation_indices), axis=-1))
        indices = np.concatenate(indices)
        
        if shuffle:
            indices = np.random.permutation(indices)

        index_dataset = tf.data.Dataset.from_tensor_slices((indices.T[0], indices.T[1]))

        self.dataset = index_dataset if dataset_transform is None else index_dataset.apply(lambda tf_dataset: dataset_transform(tf_dataset, self))

        print("dataset duration:", self.total_duration)
        print("dataset examples:", self.total_examples)
        self.max_polyphony = np.max([aa.annotation.max_polyphony for aa in self._annotated_audios])
        print("max. polyphony:", self.max_polyphony)

    def prepare_example(self, aa_index_op, annotation_index_op):
        output_types, output_shapes = zip(*[
            (tf.int16,   tf.TensorShape([self.window_size])),
            (tf.float32, tf.TensorShape([self.annotations_per_window, None])),
            (tf.float32, tf.TensorShape([self.annotations_per_window])),
            (tf.string,  None),
        ])
        outputs = tf.py_func(self._create_example, [aa_index_op, annotation_index_op], output_types)
        for output, shape in zip(outputs, output_shapes):
            output.set_shape(shape)
        return outputs
    
    def is_example_voiced(self, window, annotations, times, audio_uid):
        return tf.equal(tf.count_nonzero(tf.equal(annotations, 0)), 0)
    
    @property
    def total_duration(self):
        return sum([aa.annotation.times[-1] for aa in self._annotated_audios])
    
    @property
    def total_examples(self):
        return sum([len(aa.annotation.times)//self.annotations_per_window for aa in self._annotated_audios])

    def all_samples(self):
        samples = [aa.audio.samples for aa in self._annotated_audios]
        return np.concatenate(samples)

    def _create_example(self, aa_index, annotation_start):
        aa = self._annotated_audios[aa_index]

        annotation_end = min(len(aa.annotation.times), annotation_start + self.annotations_per_window)
        
        annotations = aa.annotation.notes[annotation_start:annotation_end]
        times = aa.annotation.times[annotation_start:annotation_end]

        len_diff = self.annotations_per_window - (annotation_end - annotation_start)
        if len_diff > 0:
            times = np.pad(times, (0, len_diff), "constant")
            annotations = np.pad(annotations, ((0, len_diff),(0,0)), "constant")

        window_start_sample = np.floor(times[0]*self.samplerate)
        audio, spectrogram = aa.audio.get_window_at_sample(window_start_sample, self.inner_window_size, self.context_width)

        return (audio, annotations, times, aa.audio.uid)

    def get_annotated_audio_by_uid(self, uid):
        for aa in self._annotated_audios:
            if aa.audio.uid == uid:
                return aa
