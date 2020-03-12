import os
from .common import melody_dataset_generator, load_melody_dataset, parallel_preload
import json

modulepath = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(modulepath, "..", "data")

prefix = "musicnet_mir"

def get_split():
    # For MusicNet we use our validation split and the extended test split used by dataset authors, see:
    # > Thickstun, John, et al. "Invariances and data augmentation for supervised music transcription."
    # > 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018.
    with open(os.path.join(modulepath, "..", "data", "musicnet_my_split.json")) as f:
        # loads the track paths splits
        split = json.load(f)
    return split

def generator():
    dataset_audio_path = os.path.join(data_root, "musicnet", "*_data")
    dataset_annot_path = os.path.join(data_root, "musicnet", "*_labels")

    return melody_dataset_generator(dataset_audio_path, dataset_annot_path)

def dataset():
    return load_melody_dataset(prefix, generator())


def prepare(preload_fn, threads=None):
    split = get_split()

    def musicnet_split_generator(name):
        return filter(lambda t: t.track_id in split[name], generator())

    train_data = load_melody_dataset(prefix, musicnet_split_generator("train"))
    test_data = load_melody_dataset(prefix, musicnet_split_generator("test"))
    valid_data = load_melody_dataset(prefix, musicnet_split_generator("validation"))

    # import warnings
    # warnings.warn("Just a subset")
    # test_data = test_data[:1]
    # train_data = train_data[:20]
    # valid_data = valid_data[:10]

    all_data = train_data+test_data+valid_data
    if threads == 1:
        for aa in all_data:
            preload_fn(aa)
    else:
        parallel_preload(preload_fn, all_data, threads=threads)
    
    # TODO: choose a better small validation
    small_validation_data = [aa.slice(0, 5) for aa in valid_data]

    return train_data, test_data, valid_data, small_validation_data
