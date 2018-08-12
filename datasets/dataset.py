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
        self.samples_count = 0
        self.samplerate = 0
        self.spectrogram_hop_size = None
        self.spectrogram = None

    def load_resampled_audio(self, samplerate):
        resampled_path = self.path.replace(".wav", "_{}.wav".format(samplerate))

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
        spec_path = self.path.replace(".wav", spec_function_thumbprint+".npy")
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
        self.frame_width = int(np.round(aa.annotation.get_frame_width()*self.samplerate))

        self.context_width = context_width
        self.annotations_per_window = annotations_per_window
        self.window_size = int(np.round(annotations_per_window*self.frame_width + 2*context_width))

        data = []

        for i, aa in enumerate(annotated_audios):
            if self.window_size > aa.audio.samples_count:
                raise RuntimeError("Window size is bigger than the audio.")

            for j in range(len(aa.annotation.times)-annotations_per_window):
                data.append((i, j))


        Dataset.__init__(self, np.array(data, dtype=np.int32), shuffle_batches)

    def all_samples(self):
        samples = [aa.audio.samples for aa in self.annotated_audios]
        return np.concatenate(samples)

    def _create_batch(self, permutation):
        batch = Dataset._create_batch(self, permutation)
        new_batch = []

        annotations = [self.annotated_audios[i].annotation.notes[j:j+self.annotations_per_window] for i,j in batch]

        max_len = max([len(annot) for annots in annotations for annot in annots])

        # transforming the batch - add actual audio snippets according to window_bounds
        for i,j in batch:
            aa = self.annotated_audios[i]

            window_start_sample = np.floor(aa.annotation.times[j]*self.samplerate)
            window_end_sample = window_start_sample + self.window_size - 2*self.context_width
            
            annotation_ragged = aa.annotation.notes[j:j+self.annotations_per_window]

            b0 = int(window_start_sample-self.context_width)
            b1 = int(window_end_sample+self.context_width)

            audio = aa.audio.get_padded_audio(b0, b1)
            spectrogram = aa.audio.get_padded_spectrogram(b0, b1)

            # padding the annotations
            annotation = [np.concatenate([annot, np.zeros(max_len - len(annot))]) for annot in annotation_ragged]

            new_batch.append({
                "audio": audio,
                "spectrogram": spectrogram,
                "annotation_ragged": annotation_ragged,
                "annotation": annotation
                })

        return new_batch