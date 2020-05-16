from . import maps
from . import Annotation
import os
import json

def test_maps_generator():
    maps_gen = list(maps.generator())
    # check file validity
    for (audio_path, annot_path, track_id) in maps_gen:
        assert os.path.isfile(audio_path)
        assert os.path.isfile(annot_path)

    # check the number of songs
    split = maps.get_split()
    split_len_sum = len(split["train"]) + len(split["test"]) + len(split["validation"])
    assert len(maps_gen) == split_len_sum

def test_maps_prepare():
    samplerate = 44100
    frame_width = 256
    # preload_fn takes each datapoint tuple and proceeds with further initialization
    def preload_fn(aa):
        annot_path, uid = aa.annotation
        aa.annotation = Annotation.from_midi(annot_path, uid, hop_samples=frame_width, unique_mf0=True)
        aa.audio.load_resampled_audio(samplerate)

    train_data, test_data, valid_data, small_validation_data = maps.prepare(preload_fn, threads=1)
    assert len(train_data) == 180
    assert len(test_data) == 60
    assert len(valid_data) == 30
    assert len(small_validation_data) == 3

    # for 


test_maps_prepare()
