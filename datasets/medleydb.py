import os
import mir_eval
import numpy as np
import json

from .common import melody_dataset_generator, load_melody_dataset, parallel_preload
modulepath = os.path.dirname(os.path.abspath(__file__))

prefix = "mdb"

def get_split():
    # For MDB and MDB synth we use the train/validation/test split according to deepsalience paper
    with open(os.path.join(modulepath, "..", "data", "mdb_ismir_split.json")) as f:
        split = json.load(f)
    return split


def generator(dataset_root, annotation_type="MELODY2"):
    dataset_audio_path = os.path.join(dataset_root, "Audio", "*")
    dataset_annot_path = os.path.join(dataset_root, "Annotations", "Melody_Annotations", annotation_type)

    return melody_dataset_generator(dataset_audio_path, dataset_annot_path, audio_suffix="_MIX.wav", annot_suffix="_"+annotation_type+".csv")


def dataset(dataset_root):
    return load_melody_dataset(prefix, generator(dataset_root))


def prepare(preload_fn, threads=None, annotation_type="MELODY2"):
    medleydb_split = get_split()

    def mdb_split(name):
        gen = generator(os.path.join(modulepath, "..", "data", "MedleyDB"), annotation_type=annotation_type)
        return filter(lambda x: x.track_id in medleydb_split[name], gen)

    train_data = load_melody_dataset(prefix, mdb_split("train"))
    test_data = load_melody_dataset(prefix, mdb_split("test"))
    valid_data = load_melody_dataset(prefix, mdb_split("validation"))

    parallel_preload(preload_fn, train_data+test_data+valid_data, threads=threads)

    # TODO: choose better small validation
    small_validation_data = [
        test_data[8].slice(20, 30),  # MatthewEntwistle_FairerHopes, harfa a nějaká flétna - vysoké tóny nezastoupené v training datech
        test_data[16].slice(53, 57),  # MusicDelta_Pachelbel, housle + violoncello, hodně souzvuků
        test_data[10].slice(6, 11),  # MatthewEntwistle_Lontano, tichý zpěv bezeslov, nad arpeggiato pianem - hodně melodický šum
        test_data[3].slice(3, 7),  # ChrisJacoby_BoothShotLincoln, akustická kytara s kytarovým podkladem
        test_data[6].slice(4, 17),  # Debussy_LenfantProdigue, mužský operní zpěv + klavír
        valid_data[9].slice(15, 21),  # HezekiahJones_BorrowedHeart, souzvuk zpěváka a zpěvačky
        valid_data[0].slice(70, 80),  # AmarLal_Rest, kytary
        valid_data[13].slice(4*60+35, 4*60+45),  # AmarLal_Rest, kytary+perkuse+basová kytara
    ]

    return train_data, test_data, valid_data, small_validation_data
