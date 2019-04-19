import evaluation
import pandas
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datasets

def load_data(paths, attributes=None, attr_names=None):
    data = []
    for attrs, path in zip(attributes, paths):
        results = evaluation.results("test", path, ".csv")
        if attributes:
            for name, attr in zip(attr_names, attrs):
                results[name] = attr
        data.append(results)
    return pandas.concat(data)


def plot_data(data, attr_names, split="MedleyDB valid.", plot_metric="Raw Pitch Accuracy", drop_metrics=['Voicing Recall', 'Voicing False Alarm', "Overall Accuracy"]):
    sns.set(rc={'figure.figsize':(14,9)})
    sns.set(style="whitegrid")

    if split is not None:
        data = data[data.Dataset==split]

    hue = None
    palette = "Blues"
    if len(attr_names) > 1:
        hue = attr_names[1]
        palette = None
    sns.boxplot(x=plot_metric, y=attr_names[0], orient="h", hue=hue, data=data, fliersize=0, palette=palette, showmeans=True, showfliers=False)
    sns.swarmplot(x=plot_metric, y=attr_names[0], orient="h", hue=hue, data=data, dodge=True, linewidth=1, edgecolor='gray', palette=palette, alpha=0.7, size=4)
    return data.drop(drop_metrics, axis=1).groupby(attr_names).mean()


def plot_note_hist(method, path, est_suffix=".csv"):
    sns.set(rc={'figure.figsize': (16, 6)})
    sns.set(style="whitegrid")

    diffs = []

    for prefix, split, dataset_name, ref_paths, est_paths in evaluation.paths_iterator(method, path, est_suffix):
        for filename, (ref_time, ref_freq, est_time, est_freq) in evaluation.load_melody_paths(ref_paths, est_paths):
            ref_notes = datasets.common.hz_to_midi_safe(ref_freq)
            est_notes = datasets.common.hz_to_midi_safe(np.abs(est_freq))
            diff = (est_notes - ref_notes)[(ref_freq > 0) & (est_freq > 0)]
            filtered_diff = diff[np.abs(diff) > 0.5]
            diffs.append(filtered_diff)
    
    fig, ax = plt.subplots()
    bins = np.arange(-24, 25)
    ax.set_ylim(0, 50000)
    ax.set_xticks(bins)
    bins = bins - 0.5
    sns.distplot(np.concatenate(diffs), kde=False, bins=bins)
