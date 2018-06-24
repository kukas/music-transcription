import soundfile as sf
import numpy as np
import os
from resampy import resample

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
        self.samplerate = 0

    def load_resampled_audio(self, samplerate):
        resampled_path = self.path.replace(".wav", "_{}.wav".format(samplerate))

        if os.path.isfile(resampled_path):
            audio, sr_orig = sf.read(resampled_path, dtype="int16")
            self.samples = audio
            self.samplerate = samplerate

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
            self.samplerate = samplerate

    def get_duration(self):
        if self.samplerate == 0:
            return 0
        return len(self.samples)/self.samplerate

    def slice(self, start, end):
        sliced = Audio(self.path, self.uid)
        sliced.samplerate = self.samplerate
        sliced.samples = self.samples[int(start*self.samplerate):int(end*self.samplerate)]
        return sliced

''' Handles the common time-frequency annotation format. '''
class Annotation:
    def __init__(self, times, notes):
        self.times = np.array(times)
        self.notes = np.array(notes)

    def get_notes(self):
        return self.times, self.notes

    def get_frame_width(self):
        if len(self.times) == 0:
            raise RuntimeError("The annotation is empty.")
        return self.times[1]-self.times[0]
    
    def slice(self, start, end):
        framerate = 1/self.get_frame_width()
        sliced_times = self.times[int(start*framerate):int(end*framerate)]
        # time offset
        sliced_times = sliced_times - sliced_times[0]
        sliced_notes = self.notes[int(start*framerate):int(end*framerate)]
        return Annotation(sliced_times, sliced_notes)

class Dataset:
    def __init__(self, data, shuffle_batches=True):
        self.data = data
        self._shuffle_batches = shuffle_batches
        self._new_permutation()

    def _new_permutation(self):
        if self._shuffle_batches:
            self._permutation = np.random.permutation(len(self.data))
        else:
            self._permutation = np.arange(len(self.data))
    
    def all_data(self):
        return self._create_batch(np.arange(len(self.data)))

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._create_batch(batch_perm)

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._new_permutation()
            return True
        return False
    
    def _create_batch(self, permutation):
        batch = np.array([self.data[i] for i in permutation])
        return batch
    
    def reset(self):
        self._new_permutation()

class AADataset(Dataset):
    def __init__(self, annotated_audios, annotations_per_window, context_width, shuffle_batches=True):
        self.annotated_audios = annotated_audios
        aa = annotated_audios[0]

        self.samplerate = aa.audio.samplerate
        self.frame_width = aa.annotation.get_frame_width()*self.samplerate

        self.context_width = context_width
        self.annotations_per_window = annotations_per_window
        self.window_size = int(np.round(annotations_per_window*self.frame_width + 2*context_width))

        data = []

        for aa in annotated_audios:
            for i in range(len(aa.annotation.times)-annotations_per_window):
                window_start_sample = np.floor(aa.annotation.times[i]*self.samplerate)
                window_end_sample = window_start_sample + self.window_size - 2*context_width

                audio_window_bounds = (int(window_start_sample-context_width), int(window_end_sample+context_width))

                window_annot = aa.annotation.notes[i:i+annotations_per_window]

                if self.window_size > len(aa.audio.samples):
                    raise RuntimeError("Window size is bigger than the audio.")

                data.append({
                    "annotaudio": aa,
                    "window_bounds": audio_window_bounds,
                    "annotation": window_annot
                    })


        Dataset.__init__(self, data, shuffle_batches)

    def all_samples(self):
        samples = [aa.audio.samples for aa in self.annotated_audios]
        return np.concatenate(samples)

    def _create_batch(self, permutation):
        batch = Dataset._create_batch(self, permutation)
        new_batch = []

        max_len = max([len(annot) for b in batch for annot in b["annotation"]])

        # transforming the batch - add actual audio snippets according to window_bounds
        for b in batch:
            b0 = cut_start = b["window_bounds"][0]
            b1 = cut_end = b["window_bounds"][1]

            samples_int16 = b["annotaudio"].audio.samples
            last_sample_index = len(samples_int16) - 1

            # padding the snippets with zeros when the context reaches outside the audio
            cut_start_diff = 0
            cut_end_diff = 0
            if b0 < 0:
                cut_start = 0
                cut_start_diff = cut_start - b0
            if b1 > last_sample_index:
                cut_end = last_sample_index
                cut_end_diff = b1 - last_sample_index

            audio = samples_int16[cut_start:cut_end]

            if cut_start_diff:
                zeros = np.zeros(cut_start_diff, dtype=np.float32)
                audio = np.concatenate([zeros, audio])

            if cut_end_diff:
                zeros = np.zeros(cut_end_diff, dtype=np.float32)
                audio = np.concatenate([audio, zeros])

            # padding the annotations
            annotation = [np.concatenate([annot, np.zeros(max_len - len(annot))]) for annot in b["annotation"]]

            new_batch.append({
                "audio": audio,
                "annotation_ragged": b["annotation"],
                "annotation": annotation
                })

        return new_batch