import mir_eval
from glob import glob
import pandas

import os
from os.path import join
import matplotlib.pyplot as plt

import csv
import datasets

from . import melody

modulepath = os.path.dirname(os.path.abspath(__file__))
mdb_split = datasets.medleydb.get_split()
wjazzd_split = datasets.wjazzd.get_split()

dataset_list = {
    "ORCHSET": (
        datasets.orchset.prefix,
        list(datasets.orchset.generator(join(modulepath, "../data/Orchset")))
        ),
    "ADC04": (
        datasets.adc2004.prefix,
        list(datasets.adc2004.generator(join(modulepath, "../data/adc2004")))
        ),
    "MIREX05": (
        datasets.mirex05.prefix,
        list(datasets.mirex05.generator(join(modulepath, "../data/mirex05")))
        ),
    "MDB-mel-s.": (
        datasets.mdb_melody_synth.prefix,
        list(filter(lambda x: x.uid in mdb_split["test"], datasets.mdb_melody_synth.generator(join(modulepath, "../data/MDB-melody-synth"))))
        ),
    "MedleyDB": (
        datasets.medleydb.prefix,
        list(filter(lambda x: x.uid in mdb_split["test"], datasets.medleydb.generator(join(modulepath, "../data/MedleyDB/MedleyDB"))))
        ),
    "WJazzD": (
        datasets.wjazzd.prefix,
        list(filter(lambda x: x.uid in wjazzd_split["test"], datasets.wjazzd.generator(join(modulepath, "../data/WJazzD"))))
        ),
    "MDB-mel-s. valid.": (
        datasets.mdb_melody_synth.prefix,
        list(filter(lambda x: x.uid in mdb_split["validation"], datasets.mdb_melody_synth.generator(join(modulepath, "../data/MDB-melody-synth"))))
    ),
    "MedleyDB valid.": (
        datasets.medleydb.prefix,
        list(filter(lambda x: x.uid in mdb_split["validation"], datasets.medleydb.generator(join(modulepath, "../data/MedleyDB/MedleyDB"))))
    ),
    "WJazzD valid.": (
        datasets.wjazzd.prefix,
        list(filter(lambda x: x.uid in wjazzd_split["validation"], datasets.wjazzd.generator(join(modulepath, "../data/WJazzD"))))
        ),
}

def evaluate_dataset_melody(refs, ests, per_track_info=False):
    refs = sorted(refs)
    ests = sorted(ests)

    all_scores = []
    for ref, est in zip(refs, ests):
        filename = os.path.splitext(os.path.basename(ref))[0]
        
        ref_time, ref_freq = mir_eval.io.load_time_series(ref, delimiter='\\s+|,')
        est_time, est_freq = mir_eval.io.load_time_series(est, delimiter='\\s+|,')
        
        scores = mir_eval.melody.evaluate(ref_time, ref_freq, est_time, est_freq)
        scores["Track"] = filename
        all_scores.append(scores)

        if per_track_info:
            plt.figure(figsize=(15, 7))
            plt.title(filename)
            plt.plot(ref_time, ref_freq, '.k', markersize=8)
            plt.plot(est_time, est_freq, '.r', markersize=3)
            plt.show()

            for k,v in scores.items():
                print(k,":",v)
            print()

    return all_scores

def summary(method, path, est_suffix):
    results = {}
    # Iterates through the datasets
    for name, (prefix, dataset_iterator) in dataset_list.items():
        # List of the paths to reference annotations
        annot_paths = map(lambda x: x.annot_path, dataset_iterator)
        audio_names = map(lambda x: os.path.splitext(os.path.basename(x.audio_path))[0], dataset_iterator)

        ests_dir = "{}-{}-melody-outputs".format(prefix, method)
        # List of the paths to estimation annotations
        est_paths = [join(path, ests_dir, name+est_suffix) for name in audio_names]

        result = evaluate_dataset_melody(annot_paths, est_paths)
        results[name] = result
    summarized = {k: pandas.DataFrame(v).mean() for k, v in results.items()}
    return pandas.DataFrame(summarized)

def evaluate_model(network, model_name):
    path = model_name+"-f0-outputs"
    if not os.path.exists(path):
        os.mkdir(path)
    # Iterates through the datasets
    for dataset_name, (prefix, dataset_iterator) in dataset_list.items():
        audios = map(lambda x: datasets.Track(x.audio_path, None, x.uid), dataset_iterator)

        ests_dir = join(path, "{}-{}-melody-outputs".format(prefix, model_name))
        if not os.path.exists(ests_dir):
            os.mkdir(ests_dir)

        print("evaluating", dataset_name)
        estimations = network.predict(audios, name=prefix)
        for uid, (est_time, est_freq) in estimations.items():
            with open(join(ests_dir, uid+".csv"), "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(list(zip(est_time, est_freq)))

    return summary(model_name, path)

