import mir_eval
from glob import glob
import pandas

import os
from os.path import join
import matplotlib.pyplot as plt

import csv
import datasets

from . import melody

import visualization as vis

modulepath = os.path.dirname(os.path.abspath(__file__))

def get_dataset_list():
    mdb_split = datasets.medleydb.get_split()
    wjazzd_split = datasets.wjazzd.get_split()

    return [
        (
            datasets.orchset.prefix,
            "test",
            "ORCHSET",
            list(datasets.orchset.generator(join(modulepath, "../data/Orchset")))
            ),
        (
            datasets.adc2004.prefix,
            "test",
            "ADC04",
            list(datasets.adc2004.generator(join(modulepath, "../data/adc2004")))
            ),
        (
            datasets.mirex05.prefix,
            "test",
            "MIREX05",
            list(datasets.mirex05.generator(join(modulepath, "../data/mirex05")))
            ),
        (
            datasets.mdb_melody_synth.prefix,
            "test",
            "MDB-mel-s.",
            list(filter(lambda x: x.track_id in mdb_split["test"], datasets.mdb_melody_synth.generator(join(modulepath, "../data/MDB-melody-synth"))))
            ),
        (
            datasets.medleydb.prefix,
            "test",
            "MedleyDB",
            list(filter(lambda x: x.track_id in mdb_split["test"], datasets.medleydb.generator(join(modulepath, "../data/MedleyDB/MedleyDB"))))
            ),
        (
            datasets.wjazzd.prefix,
            "test",
            "WJazzD",
            list(filter(lambda x: x.track_id in wjazzd_split["test"], datasets.wjazzd.generator(join(modulepath, "../data/WJazzD"))))
            ),
        (
            datasets.mdb_melody_synth.prefix,
            "valid",
            "MDB-mel-s. valid.",
            list(filter(lambda x: x.track_id in mdb_split["validation"], datasets.mdb_melody_synth.generator(join(modulepath, "../data/MDB-melody-synth"))))
        ),
        (
            datasets.medleydb.prefix,
            "valid",
            "MedleyDB valid.",
            list(filter(lambda x: x.track_id in mdb_split["validation"], datasets.medleydb.generator(join(modulepath, "../data/MedleyDB/MedleyDB"))))
        ),
        (
            datasets.wjazzd.prefix,
            "valid",
            "WJazzD valid.",
            list(filter(lambda x: x.track_id in wjazzd_split["validation"], datasets.wjazzd.generator(join(modulepath, "../data/WJazzD"))))
        ),
    ]

def evaluate_dataset_melody(refs, ests, per_track_info=False):
    refs = sorted(refs)
    ests = sorted(ests)

    all_scores = []
    for ref, est in zip(refs, ests):
        filename = os.path.splitext(os.path.basename(ref))[0]

        if not os.path.exists(ref) or not os.path.exists(est):
            if all_scores != []:
                raise ValueError("Estimation or reference files not complete.")
            continue
        
        ref_time, ref_freq = mir_eval.io.load_time_series(ref, delimiter='\\s+|,')
        est_time, est_freq = mir_eval.io.load_time_series(est, delimiter='\\s+|,')
        
        scores = mir_eval.melody.evaluate(ref_time, ref_freq, est_time, est_freq)
        scores["Track"] = filename
        all_scores.append(scores)

        if per_track_info:
            est_voicing = est_freq > 0
            est_freq, est_voicing = mir_eval.melody.resample_melody_series(est_time, est_freq, est_voicing, ref_time, "linear")

            ref = datasets.common.melody_to_multif0(datasets.common.hz_to_midi_safe(ref_freq))
            est = datasets.common.melody_to_multif0(datasets.common.hz_to_midi_safe(est_freq))
            vis.draw_notes(ref, est, dynamic_figsize=False)
            plt.show()

            for k,v in scores.items():
                print(k,":",v)
            print()

    return all_scores

def results(method, path, est_suffix=".csv"):
    results = []
    # Iterates through the datasets
    for (prefix, split, dataset_name, dataset_iterator) in get_dataset_list():
        # List of the paths to reference annotations
        annot_paths = map(lambda x: x.annot_path, dataset_iterator)
        audio_names = map(lambda x: os.path.splitext(os.path.basename(x.audio_path))[0], dataset_iterator)

        ests_dir = "{}-{}-melody-outputs".format(prefix, method)
        if os.path.exists(join(path, ests_dir)):
            # List of the paths to estimation annotations
            est_paths = [join(path, ests_dir, name+est_suffix) for name in audio_names]
            if not all(os.path.exists(path) for path in est_paths):
                continue

            est_last_access = max(os.path.getmtime(path) for path in est_paths)
            saved_results_path = join(path, "eval-{}_{}-{}.pkl".format(prefix, split, est_last_access))
            if os.path.exists(saved_results_path):
                result = pandas.read_pickle(saved_results_path)
                results.append(result)
            else:
                result = evaluate_dataset_melody(annot_paths, est_paths)
                if result:
                    result = pandas.DataFrame(result)
                    result["Prefix"] = prefix
                    result["Split"] = split
                    result["Dataset"] = dataset_name
                    results.append(result)

                    result.to_pickle(saved_results_path)
    if not results:
        return pandas.DataFrame()
    return pandas.concat(results)

def summary(method, path, est_suffix=".csv"):
    data = results(method, path, est_suffix)
    return data.groupby(["Dataset"]).mean().transpose()
