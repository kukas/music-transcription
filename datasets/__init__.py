from . import musicnet

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
        self.audio = False
        self.samplerate = 0

    def load_resampled_audio(self, samplerate):
        # audio, samplerate = sf.read(self.path)
        resampled_path = self.path.replace(".wav", "_{}.wav".format(samplerate))

        if os.path.isfile(resampled_path):
            audio, sr_orig = sf.read(resampled_path)
            self.audio = audio.astype(np.float32)
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

            self.audio = audio_low
            self.samplerate = samplerate

    def get_spectrogram(self):
        pass

    def get_duration(self):
        return len(self.audio)/self.samplerate

''' Handles the common time-frequency annotation format. '''
class Annotation:
    def __init__(self, times, notes):
        self.times = np.array(times)
        self.notes = np.array(notes)

    def get_notes(self):
        return self.times, self.notes

class Dataset:
    def __init__(self, annotated_audios):
        self.annotated_audios = annotated_audios

    def get_annotated_audio_windows(self, annotations_per_window, context_width):
        # if not self.annotated_audios:
        #     raise RuntimeError("The dataset is empty.")

        # aa = self.annotated_audios[0]
        pass
