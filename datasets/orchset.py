import os
from .common import melody_dataset_generator, load_melody_dataset, parallel_preload

modulepath = os.path.dirname(os.path.abspath(__file__))

prefix = "orchset"


def generator(dataset_root):
    dataset_audio_path = os.path.join(dataset_root, "audio", "mono")
    dataset_annot_path = os.path.join(dataset_root, "GT")

    return melody_dataset_generator(dataset_audio_path, dataset_annot_path, annot_suffix=".mel")


def dataset(dataset_root):
    return load_melody_dataset(prefix, generator(dataset_root))


def prepare(preload_fn, threads=None):
    test_data = dataset(os.path.join(modulepath, "..", "data", "Orchset"))

    parallel_preload(preload_fn, test_data, threads=threads)

    small_validation_data = [
        test_data[41],  # Profofiev-Romeo&Juliet-DanceKnights-ex2, nejdřív melodii drží zeště a smyčce jsou nad nimi, pak jsou zeště doprovod
        test_data[45],  # Rimski-Korsakov-Scheherazade-Kalender-ex1, flétny vs. zeště + smyčce dělají rázy
        test_data[1],  # Beethoven-S3-I-ex2, dlouhé tóny jsou doprovod, melodie je rytmicky pravidelná, ale rychlá
        test_data[37].slice(0, 5),  # Musorgski-Ravel-PicturesExhibition-ex6, největší octave error, melodie je jasná, navíc jsou jen tympány, ale algoritmus by měl vybrat nejvyšší hrající tón
    ]

    return test_data, small_validation_data
