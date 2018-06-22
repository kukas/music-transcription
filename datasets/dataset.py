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

''' Audio container
Audio can be represented in different forms - as raw audio with various
sampling rates or as spectrograms. '''
class Audio:
    def __init__(self, path, uid):
        self.path = path
        self.uid = uid
        self.samples = False
        self.samplerate = 0

    def load_resampled_audio(self, samplerate):
        # audio, samplerate = sf.read(self.path)
        resampled_path = self.path.replace(".wav", "_{}.wav".format(samplerate))

        if os.path.isfile(resampled_path):
            audio, sr_orig = sf.read(resampled_path)
            self.samples = audio.astype(np.float32)
            self.samplerate = samplerate
            
            print(self.uid+"_"+str(samplerate), "{:.2f} min".format(self.get_duration()/60))

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

    def get_spectrogram(self):
        pass

    def get_duration(self):
        return len(self.samples)/self.samplerate

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
    def get_annotated_audio_windows(self, annotations_per_window, context_width):
        if len(self.annotated_audios) == 0:
            raise RuntimeError("The dataset is empty.")

        aa = self.annotated_audios[0]

        samplerate = aa.audio.samplerate
        frame_width = aa.annotation.get_frame_width()

        window_sample_count = annotations_per_window*frame_width + 2*context_width

    def __init__(self, annotated_audios, annotations_per_window, context_width, shuffle_batches=True):
        aa = annotated_audios[0]

        samplerate = aa.audio.samplerate
        frame_width = aa.annotation.get_frame_width()*samplerate

        window_sample_count = np.round(annotations_per_window*frame_width) + 2*context_width

        data = []

        for aa in annotated_audios:
            # zeros = np.zeros(context_width, dtype=np.float32)

            for i in range(len(aa.annotation.times)-annotations_per_window):
                window_start_sample = np.floor(aa.annotation.times[i]*samplerate)
                window_end_sample = np.floor(aa.annotation.times[i+annotations_per_window]*samplerate)

                audio_window_bounds = (int(window_start_sample-context_width), int(window_end_sample+context_width))

                window_annot = aa.annotation.notes[i:i+annotations_per_window]

                data.append({
                    "annotaudio": aa,
                    "window_bounds": audio_window_bounds,
                    "annotation": window_annot
                    })


        Dataset.__init__(self, data, shuffle_batches)

    def _create_batch(self, permutation):
        batch = Dataset._create_batch(self, permutation)
        batch = np.array([{
            "audio": b["annotaudio"].audio.samples[b["window_bounds"][0]:b["window_bounds"][1]],
            "annotation": b["annotation"]
            } for b in batch])
        return batch