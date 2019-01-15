from .musicnet import musicnet_dataset
# from .orchset import orchset_dataset
# from .adc2004 import adc2004_dataset
# from .mirex05 import mirex05_dataset
# from .medleydb import medleydb_dataset
# from .mdb_melody_synth import *
# import mdb_melody_synth

def load_all(data, samplerate):
    for i,d in enumerate(data):
        d.audio.load_resampled_audio(samplerate)
        print(".", end=("" if (i+1) % 20 else "\n"))
