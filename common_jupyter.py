import evaluation
import pandas
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datasets
import re
from glob import glob

def load_data(paths, attributes=None, attr_names=None):
    data = []
    for attrs, path in zip(attributes, paths):
        results = evaluation.results("test", path, ".csv")
        if attributes:
            for name, attr in zip(attr_names, attrs):
                results[name] = attr
        data.append(results)
    return pandas.concat(data)


def ld(attr_names, attr_types, regex, experiments_dir, verbose=False):
    experiments_paths = get_paths(experiments_dir)
    paths = list(filter(lambda x: re.search(regex, x), experiments_paths))
    attributes = get_attrs_from_paths(regex, attr_types, paths)
    if verbose:
        a = map(str, zip([p.split("/")[-2] for p in paths], attributes))
        print("\n".join(a))
    return load_data(paths, attributes, attr_names)


def get_attrs_from_paths(regex, types, paths):
    out = []
    #for attr_re, type_fn in zip(attr_re_list, types):
    for path in paths:
        values = re.findall(regex, path)
        assert len(values) == 1
        values = values[0]
        if not isinstance(values, tuple):
            values = (values,)
        out.append([type_fn(v) for v, type_fn in zip(values, types)])

    return out

def to_latex(df):
    df = df.reset_index()
    # print(df.columns.values.tolist())
    aliases = {"Raw Pitch Accuracy": "RPA", "Raw Chroma Accuracy": "RCA", "Voicing Accuracy": "VA", "Overall Accuracy": "OA", "Voicing False Alarm": "VFA", "Voicing Recall": "VR"}
    header = map(lambda x: aliases[x] if x in aliases else x, df.columns.values)
    print(df.to_latex(float_format=lambda x: "%.3f"%x, bold_rows=True, header=list(header), index=False))


def plot_grid(data, attr_names, name="test", split="MedleyDB valid.", axs=None, vminRPA=None, vmaxRPA=None, vminRCA=None, vmaxRCA=None):
    if axs is None:
        sns.set(rc={'figure.figsize': (10, 4)})
        sns.set(style="whitegrid")
        fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
        axs[1].set_ylabel("")
        plt.tight_layout()

    cmap = sns.cubehelix_palette(100, reverse=True, as_cmap=True)

    pivot = data[data.Dataset==split].groupby(attr_names).mean().reset_index().pivot(attr_names[0], attr_names[1], "Raw Pitch Accuracy")
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, ax=axs[0], vmin=vminRPA, vmax=vmaxRPA)

    pivot = data[data.Dataset==split].groupby(attr_names).mean().reset_index().pivot(attr_names[0], attr_names[1], "Raw Chroma Accuracy")
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, ax=axs[1], vmin=vminRCA, vmax=vmaxRCA)


    if axs is None:
        fig.savefig("figures/"+name+".pdf", bbox_inches="tight")
    return pivot

def plot_data(data, attr_names, name="test", split="MedleyDB valid.", plot_metric="Raw Pitch Accuracy", palette="cubehelix", drop_metrics=["Voicing Accuracy",'Voicing Recall', 'Voicing False Alarm', "Overall Accuracy"], figsize=(8, None), order=None):

    if split is not None:
        data = data[data.Dataset==split]

    hue = None
    # palette = sns.cubehelix_palette(8)
    num_bars = len(data.groupby(attr_names))
    categories = data[attr_names[0]].unique()

    if len(attr_names) > 1:
        hue = attr_names[1]
        categories = data[attr_names[1]].unique()
    #     palette = sns.cubehelix_palette(8)

    if palette == "cubehelix":
        palette = sns.cubehelix_palette(len(categories)+2)
    if figsize[1] is None:
        figsize = (figsize[0], num_bars*0.5)

    sns.set(rc={'figure.figsize': figsize})
    sns.set(style="whitegrid")

    _order = None
    if order:
        _order = data.groupby(attr_names).mean().reset_index().sort_values(plot_metric)[attr_names[0]]
    ax = sns.boxplot(x=plot_metric, y=attr_names[0], orient="h", hue=hue, data=data, fliersize=2, palette=palette, showmeans=True, showfliers=False,
                     meanprops={"markerfacecolor": "black", "markeredgecolor": "black"}, order=_order)
    # sns.swarmplot(x=plot_metric, y=attr_names[0], orient="h", hue=hue, data=data, dodge=True, linewidth=1, edgecolor='gray', palette=palette, alpha=0.7, size=4)

    figure = ax.get_figure()
    figure.savefig("figures/"+name+".pdf", bbox_inches="tight")

    summary = data.drop(drop_metrics, axis=1).groupby(attr_names).mean()
    if order:
        summary = summary.sort_values(plot_metric)
    return summary


def plot_note_hist(method, path, path2, est_suffix=".csv"):
    sns.set(rc={'figure.figsize': (8, 6)})
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
    bins = np.arange(-5, 5)
    ax.set_ylim(0, 50000)
    ax.set_xticks(bins)
    bins = bins - 0.5
    sns.distplot(np.concatenate(diffs), kde=False, bins=bins)


def get_paths(path):
    paths = sorted(glob(path+"/*/model-f0-outputs") + glob(path+"/*/*/model-f0-outputs"))
    return list(filter(lambda x: "koš" not in x, paths))  # vyhoď modely v koši
