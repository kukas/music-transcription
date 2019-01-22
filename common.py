
import json
import datasets
import datetime

def get_medley_split():
    # For MDB and MDB synth we use the train/validation/test split according to deepsalience paper
    with open("data/MedleyDB/dataset_ismir_split.json") as f:
        medley_split = json.load(f)
    return medley_split

def prepare_medley():
    medley_split = get_medley_split()

    def mdb_split(name):
        gen = datasets.medleydb.generator("data/MedleyDB/MedleyDB/")
        return filter(lambda x: x.uid in medley_split[name], gen)

    train_data = datasets.load_melody_dataset(datasets.medleydb.prefix, mdb_split("train"))
    valid_data = datasets.load_melody_dataset(datasets.medleydb.prefix, mdb_split("validation"))
    small_validation_data = [
        valid_data[3].slice(15, 20.8),
        valid_data[9].slice(56, 61.4),
        valid_data[5].slice(55.6, 61.6),
    ]

    return train_data, valid_data, small_validation_data

def prepare_mdb_synth_stems():
    medley_split = get_medley_split()

    def mdb_split(name):
        gen = datasets.mdb_stem_synth.generator("data/MDB-stem-synth/")
        return filter(lambda x: x.uid[:-len("_STEM_xx")] in medley_split[name], gen)

    train_data = datasets.load_melody_dataset(datasets.mdb_stem_synth.prefix, mdb_split("train"))
    valid_data = datasets.load_melody_dataset(datasets.mdb_stem_synth.prefix, mdb_split("validation"))
    small_validation_data = [
        valid_data[3].slice(0, 40),
    ]

    return train_data, valid_data, small_validation_data


def name(args, prefix=""):
    name = "{}-{}-bs{}-apw{}-fw{}-ctx{}-nr{}-sr{}".format(
        prefix,
        datetime.datetime.now().strftime("%m-%d_%H%M%S"),
        args["batch_size"],
        args["annotations_per_window"],
        args["frame_width"],
        args["context_width"],
        args["note_range"],
        args["samplerate"],
    )
    args["logdir"] = "models/" + name
    
    return name
