from .dataset import AnnotatedAudio, Audio, Annotation, AADataset
from . import common
from . import adc2004
from . import mirex05
from . import orchset
from . import medleydb
from . import mdb_melody_synth
from . import mdb_stem_synth

from .common import load_melody_dataset, Track

def load_all(data, samplerate):
    for i,d in enumerate(data):
        d.audio.load_resampled_audio(samplerate)
        print(".", end=("" if (i+1) % 20 else "\n"))
    print()