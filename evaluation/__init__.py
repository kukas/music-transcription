import mir_eval
from glob import glob
import pandas

import os
from os.path import join
import matplotlib.pyplot as plt

# Evaluujeme pouze podmnožinu MedleyDB a MDB-melody-synth
modulepath = os.path.dirname(__file__)
import json
with open(join(modulepath, "../data/MedleyDB/dataset_ismir_split.json")) as f:
    medley_test_subset = json.load(f)["test"]

import datasets

dataset_list = {
    "ORCHSET": (
        "orchset",
        list(datasets.orchset.generator(join(modulepath, "../data/Orchset")))
        ),
    "ADC04": (
        "adc04",
        list(datasets.adc2004.generator(join(modulepath, "../data/adc2004")))
        ),
    "MIREX05": (
        "mirex05",
        list(datasets.mirex05.generator(join(modulepath, "../data/mirex05")))
        ),
    "MDB-f0-s.": (
        "mdb-melody-synth_test",
        list(filter(lambda x: x.uid in medley_test_subset, datasets.mdb_melody_synth.generator(join(modulepath, "../data/MDB-melody-synth"))))
        ),
    "MedleyDB": (
        "medleydb_test",
        list(filter(lambda x: x.uid in medley_test_subset, datasets.medleydb.generator(join(modulepath, "../data/MedleyDB/MedleyDB"))))
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

def summary(method, path):
    results = {}

    for name, (prefix, dataset_iterator) in dataset_list.items():
        refs = map(lambda x: x.annot_path, dataset_iterator)

        ests_dir = "{}-{}-melody-outputs".format(prefix, method)
        ests = glob(join(path, ests_dir, "*"))

        result = evaluate_dataset_melody(refs, ests)
        results[name] = result
    summarized = {k: pandas.DataFrame(v).mean() for k, v in results.items()}
    return pandas.DataFrame(summarized)
