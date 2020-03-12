import numpy as np
import os
import librosa
import tensorflow as tf

import datasets

from .annotatedaudio import AnnotatedAudio
from .annotation import Annotation
from .audio import Audio

np.random.seed(42)

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
                raise RuntimeError("Window size is bigger than the audio.\nwindow_size={}, aa.audio.samples_count={}".format(self.window_size, aa.audio.samples_count))

            annot_length = len(aa.annotation.times)
            annotation_indices = np.arange(0, annot_length, self.hop_size, dtype=np.int32)
            aa_indices = np.full((len(annotation_indices),), aa_index, dtype=np.int32)

            indices.append(np.stack((aa_indices, annotation_indices), axis=-1))
        indices = np.concatenate(indices)
        
        if shuffle:
            indices = np.random.permutation(indices)

        index_dataset = tf.data.Dataset.from_tensor_slices((indices.T[0], indices.T[1]))

        self.output_types, self.output_shapes = list(zip(*[
            (tf.int16,   tf.TensorShape([self.window_size])), # audio
            (tf.uint16,    tf.TensorShape([None, None, None])), # spectrogram
            (tf.float32, tf.TensorShape([self.annotations_per_window, None])), # annotations
            (tf.float64, tf.TensorShape([self.annotations_per_window])), # times
            (tf.string,  None), # uid
        ]))

        self.dataset = index_dataset if dataset_transform is None else index_dataset.apply(lambda tf_dataset: dataset_transform(tf_dataset, self))

        print("dataset id:", _annotated_audios[0].audio.uid.split("_")[0])
        print("dataset duration: {:.2f} minutes".format(self.total_duration/60))
        print("dataset examples:", self.total_examples)
        self.max_polyphony = np.max([aa.annotation.max_polyphony for aa in self._annotated_audios])
        print("max. polyphony:", self.max_polyphony)
        print("max. note:", np.max([np.max(aa.annotation.notes) for aa in self._annotated_audios]))
        print("min. note:", np.min([np.min(aa.annotation.notes) for aa in self._annotated_audios]))
        if self.annotations_per_window != self.hop_size:
            print("using hop_size", self.hop_size)
        print()

    def prepare_example(self, aa_index_op, annotation_index_op):
        outputs = tf.py_func(self._create_example, [aa_index_op, annotation_index_op], self.output_types)
        for output, shape in zip(outputs, self.output_shapes):
            output.set_shape(shape)
        return outputs
    
    def is_example_voiced(self, window_op, spectrogram_op, annotations_op, times_op, audio_uid_op):
        return tf.equal(tf.count_nonzero(tf.equal(annotations_op, 0)), 0)
    
    def mix_example_with(self, audio):
        def _mix_example(window_op, spectrogram_op, annotations_op, times_op, audio_uid_op):
            def mix_with(window, spectrogram, annotations, times, audio_uid):
                window = (window + audio[:len(window)])//2
                return (window, spectrogram, annotations, times, audio_uid)

            outputs = tf.py_func(mix_with, [window_op, annotations_op, times_op, audio_uid_op], self.output_types)
            for output, shape in zip(outputs, self.output_shapes):
                output.set_shape(shape)
            return outputs
        return _mix_example
    
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
        if annotations.shape[1] < self.max_polyphony:
            annotations = np.pad(annotations, ((0, 0), (0, self.max_polyphony - annotations.shape[1])), "constant", constant_values=-1)

        times = aa.annotation.times[annotation_start:annotation_end]

        len_diff = self.annotations_per_window - (annotation_end - annotation_start)
        if len_diff > 0:
            times = np.pad(times, (0, len_diff), "constant", constant_values=(-1, -1))
            annotations = np.pad(annotations, ((0, len_diff), (0, 0)), "constant", constant_values=-1)

        window_start_sample = int(np.round(times[0]*self.samplerate))
        audio, spectrogram = aa.audio.get_window_at_sample(window_start_sample, self.inner_window_size, self.context_width)
        if audio is None:
            audio = np.zeros((0, 0), dtype=np.int16)
        if spectrogram is None:
            spectrogram = np.zeros((0, 0, 0, 0), dtype=np.uint16)

        if len(spectrogram.shape) == 2:
            spectrogram = np.expand_dims(spectrogram, -1)
        
        return (audio, spectrogram, annotations, times, aa.audio.uid)

    def get_annotated_audio_by_uid(self, uid):
        for aa in self._annotated_audios:
            if aa.audio.uid == uid:
                return aa

    # # Step 2 : Frequency masking
    # for i in range(frequency_mask_num):
    #     f = np.random.uniform(low=0.0, high=frequency_masking_para)
    #     f = int(f)
    #     f0 = random.randint(0, v - f)
    #     warped_mel_spectrogram[f0:f0 + f, :] = 0

    # # Step 3 : Time masking
    # for i in range(time_mask_num):
    #     t = np.random.uniform(low=0.0, high=time_masking_para)
    #     t = int(t)
    #     t0 = random.randint(0, tau - t)
    #     warped_mel_spectrogram[:, t0:t0 + t] = 0
    # https://github.com/shelling203/SpecAugment/blob/master/SpecAugment/spec_augment_tensorflow.py
    def specaugment(self, args):
        def _mix_example(window_op, spectrogram_op, annotations_op, times_op, audio_uid_op):
            def _specaugment(window, spectrogram, annotations, times, audio_uid):
                if np.random.uniform() < args.specaugment_prob:
                    _, v, tau = spectrogram.shape
                    # Frequency masking
                    for _ in range(args.specaugment_freq_mask_num):
                        f = np.random.randint(1, args.specaugment_freq_mask_max)
                        f0 = np.random.randint(0, v - f)
                        spectrogram[:, f0:f0+f, :] = 0
                    # Time masking
                    for _ in range(args.specaugment_time_mask_num):
                        t = np.random.randint(1, args.specaugment_time_mask_max)
                        t0 = np.random.randint(0, tau - t)
                        spectrogram[:, :, t0:t0+t] = 0

                return (window, spectrogram, annotations, times, audio_uid)

            outputs = tf.py_func(_specaugment, [window_op, spectrogram_op, annotations_op, times_op, audio_uid_op], self.output_types)
            for output, shape in zip(outputs, self.output_shapes):
                output.set_shape(shape)
            return outputs
        return _mix_example
