
import json
import datasets
import datetime

def prepare_medley():
    # For MDB and MDB synth we use the train/validation/test split according to deepsalience paper
    with open("data/MedleyDB/dataset_ismir_split.json") as f:
        medley_split = json.load(f)

    def gen_mdb(): return datasets.medleydb.generator("data/MedleyDB/MedleyDB/")

    train_data = datasets.load_melody_dataset(datasets.medleydb.prefix, filter(lambda x: x.uid in medley_split["train"], gen_mdb()))
    valid_data = datasets.load_melody_dataset(datasets.medleydb.prefix, filter(lambda x: x.uid in medley_split["validation"], gen_mdb()))
    # small dataset for manual evaluation
    small_validation_data = [
        valid_data[3].slice(15, 20.8),
        valid_data[9].slice(56, 61.4),
        valid_data[5].slice(55.6, 61.6),
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
