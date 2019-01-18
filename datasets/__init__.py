from .musicnet import musicnet_dataset
from .dataset import AnnotatedAudio, Audio, Annotation, AADataset

def load_all(data, samplerate):
    for i,d in enumerate(data):
        d.audio.load_resampled_audio(samplerate)
        print(".", end=("" if (i+1) % 20 else "\n"))
    print()