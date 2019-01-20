import soundfile as sf
import numpy as np
import os
from resampy import resample
import librosa
import tensorflow as tf

PROCESSED_FILES_PATH = "./processed"

def check_processed_dir():
    if not os.path.isdir(PROCESSED_FILES_PATH):
        os.makedirs(PROCESSED_FILES_PATH)

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
        self.uid = uid
        self.samples = []
        self.samples_count = 0
        self.samplerate = 0
        self.spectrogram_hop_size = None
        self.spectrogram = None

    def load_resampled_audio(self, samplerate):
        check_processed_dir()
        resampled_path = os.path.join(PROCESSED_FILES_PATH, self.uid+"_{}.wav".format(samplerate))

        if os.path.isfile(resampled_path):
            audio, sr_orig = sf.read(resampled_path, dtype="int16")
            self.samples = audio

            assert sr_orig == samplerate
        else:
            print("resampling", self.uid)
            audio, sr_orig = sf.read(self.path)
            print(audio.shape)
                    
            if len(audio.shape) >= 2: # mono downmixing, if needed
                audio = np.mean(audio, axis=1)
            audio_low = resample(audio, sr_orig, samplerate)

            sf.write(resampled_path, audio_low.astype(np.float32), samplerate)

            self.samples = audio_low

        self.samples_count = len(self.samples)
        self.samplerate = samplerate
    
    def load_spectrogram(self, spec_function, spec_function_thumbprint, hop_size):
        self.spectrogram_hop_size = hop_size

        check_processed_dir()
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
            zeros = np.zeros(cut_start_diff, dtype=np.float32)
            audio = np.concatenate([zeros, audio])

        if cut_end_diff:
            zeros = np.zeros(cut_end_diff, dtype=np.float32)
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
        sliced = Audio(self.path, self.uid)
        sliced.samplerate = self.samplerate
        b0, b1 = int(start*self.samplerate), int(end*self.samplerate)
        sliced.samples = self.samples[b0:b1]
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
    def __init__(self, times, freqs=None, notes=None):
        assert not (freqs is None and notes is None)

        self.times = np.array(times)
        self.freqs = freqs
        self.notes = notes

        if freqs is None:
            self.freqs = [librosa.core.midi_to_hz(np.array(notes_frame)) for notes_frame in notes]

        if notes is None:
            self.notes = [librosa.core.hz_to_midi(np.array(freqs_frame)) for freqs_frame in freqs]
        
        self.max_polyphony = np.max([len(notes_frame) for notes_frame in self.notes])

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
        return Annotation(sliced_times, sliced_freqs, sliced_notes)

class AADataset:
    def __init__(self, _annotated_audios, args, preload_hook=None, dataset_transform=None):
        self._annotated_audios = _annotated_audios

        # self.frame_width = int(np.round(aa.annotation.get_frame_width()*self.samplerate))
        self.frame_width = args["frame_width"]

        self.context_width = args["context_width"]
        self.annotations_per_window = args["annotations_per_window"]
        # todo: přejmenovat na window_width?
        self.inner_window_size = self.annotations_per_window*self.frame_width
        self.window_size = self.inner_window_size + 2*self.context_width

        if preload_hook is not None:
            for aa in self._annotated_audios:
                preload_hook(aa)

                if aa.annotation is None:
                    # add dummy annotation if the annotation is missing
                    times = np.arange(0, aa.audio.samples_count, self.frame_width) / aa.audio.samplerate
                    freqs = np.tile(440, [len(times),1])
                    notes = np.tile(69, [len(times),1])

                    aa.annotation = Annotation(times, freqs, notes)

        self.samplerate = _annotated_audios[0].audio.samplerate

        output_types, output_shapes = zip(*[
            (tf.int16,   tf.TensorShape([self.window_size])),
            (tf.int32,   tf.TensorShape([self.annotations_per_window, None])),
            (tf.float32, tf.TensorShape([self.annotations_per_window])),
            (tf.string,  None),
        ])

        dataset = tf.data.Dataset.from_generator(self._generator, output_types, output_shapes)

        self.dataset = dataset if dataset_transform is None else dataset_transform(dataset)

    @property
    def total_duration(self):
        total_duration = 0
        for aa in self._annotated_audios:
            total_duration += aa.annotation.times[-1]
        return total_duration

    def _generator(self):
        for aa in self._annotated_audios:
            if self.window_size > aa.audio.samples_count:
                raise RuntimeError("Window size is bigger than the audio.")

            for annotation_index in range(0, len(aa.annotation.times)-self.annotations_per_window, self.annotations_per_window):
                yield self._create_example(aa, annotation_index)

    def all_samples(self):
        samples = [aa.audio.samples for aa in self._annotated_audios]
        return np.concatenate(samples)

    def _create_example(self, aa, annotation_start):
        annotation_end = annotation_start + self.annotations_per_window
        annotations_ragged = aa.annotation.notes[annotation_start:annotation_end]
        annotations_ragged = np.round(annotations_ragged)
        annotations = [np.concatenate([annot, np.zeros(aa.annotation.max_polyphony - len(annot))]) for annot in annotations_ragged]
        times = aa.annotation.times[annotation_start:annotation_end]

        window_start_sample = np.floor(times[0]*self.samplerate)
        audio, spectrogram = aa.audio.get_window_at_sample(window_start_sample, self.inner_window_size, self.context_width)

        return (audio, annotations, times, aa.audio.uid)

    def get_annotated_audio_by_uid(self, uid):
        for aa in self._annotated_audios:
            if aa.audio.uid == uid:
                return aa
