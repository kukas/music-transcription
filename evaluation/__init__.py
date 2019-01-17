import mir_eval
from glob import glob
import pandas
import os
import matplotlib.pyplot as plt

# Evaluujeme pouze podmnožinu MedleyDB a MDB-melody-synth
modulepath = os.path.dirname(__file__)
import json
with open(os.path.join(modulepath, "../datasets/MedleyDB/dataset_ismir_split.json")) as f:
    medley_test_subset = json.load(f)["test"]

datasets = {
    "ORCHSET": ("orchset", os.path.join(modulepath, "../datasets/Orchset/GT/*.mel"), None),
    "ADC04": ("adc04", os.path.join(modulepath, "../datasets/adc2004/adc2004_full_set/*REF.txt"), None),
    "MIREX05": ("mirex05", os.path.join(modulepath, "../datasets/mirex05/mirex05TrainFiles/*REF.txt"), None),
    "MDB-f0-s.": ("mdb-melody-synth_test", os.path.join(modulepath, "../datasets/MDB-melody-synth/annotation_melody/*.csv"), medley_test_subset),
    "MedleyDB": ("medleydb_test", os.path.join(modulepath, "../datasets/MedleyDB/MedleyDB/Annotations/Melody_Annotations/MELODY2/*.csv"), medley_test_subset),
}

def evaluate_dataset_melody(refs, ests, subset=None, per_track_info=False):
    refs_all = sorted(refs)
    refs = []
    if subset:
        for ref in refs_all:
            if any([name in ref for name in subset]):
                refs.append(ref)
    else:
        refs = refs_all

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
