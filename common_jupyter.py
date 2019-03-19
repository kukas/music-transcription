import evaluation
import pandas
import os
import seaborn as sns

def load_data(paths, attributes=None, attr_names=None):
    data = []
    for attrs, path in zip(attributes, paths):
        results = evaluation.results("test", path, ".csv")
        if attributes:
            for name, attr in zip(attr_names, attrs):
                results[name] = attr
        data.append(results)
    return pandas.concat(data)

def plot_data(data, attr_names):
    sns.set(rc={'figure.figsize':(10,6)})
    sns.set(style="whitegrid")

    data = data[data.Dataset=="MedleyDB"]
    plot_metric = "Raw Pitch Accuracy"
    drop_metrics = ['Voicing Recall', 'Voicing False Alarm', "Overall Accuracy"]

    hue = None
    palette = "Blues"
    if len(attr_names) > 1:
        hue = attr_names[1]
        palette = None
    sns.boxplot(x=plot_metric, y=attr_names[0], hue=hue, data=data, fliersize=0, palette=palette, showmeans=True)
    sns.swarmplot(x=plot_metric, y=attr_names[0], hue=hue, data=data, dodge=True, linewidth=1, edgecolor='gray', palette=palette, alpha=0.7, size=4)
    return data.drop(drop_metrics, axis=1).groupby(attr_names).mean()
