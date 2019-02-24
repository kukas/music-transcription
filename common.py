import tensorflow as tf
import json
import datasets
import datetime
from model import VD, VisualOutputHook, MetricsHook, MetricsHook_mf0, VisualOutputHook_mf0, SaveBestModelHook

def name(args, prefix=""):
    if args.logdir is None:
        filtered = ["threads", "logdir"]
        prefix = "crepe"
        name = "{}-{}".format(datetime.datetime.now().strftime("%m%d_%H%M%S"), prefix)
        for k, v in vars(args).items():
            if k not in filtered:
                short_k = "".join([w[0] for w in k.split("_")])
                name += "-{}{}".format(short_k, v)

        args.logdir = "models/" + name

        print()
        print(name)
        print()


def bn_conv(inputs, filters, size, strides, padding, activation=None, dilation_rate=1, training=False):
    name = "bn_conv{}-f{}-s{}-dil{}-{}".format(size, filters, strides, dilation_rate, padding)
    with tf.name_scope(name):
        l = tf.layers.conv1d(inputs, filters, size, strides, padding, activation=None, use_bias=False, dilation_rate=dilation_rate)
        l = tf.layers.batch_normalization(l, training=training)
        if activation:
            return activation(l)
        else:
            return l


def conv(inputs, filters, size, strides, padding, activation=None, dilation_rate=1, training=False):
    name = "conv{}-f{}-s{}-dil{}-{}".format(size, filters, strides, dilation_rate, padding)
    return tf.layers.conv1d(inputs, filters, size, strides, padding, activation=activation, dilation_rate=dilation_rate, name=name)


def prepare_datasets(which, args, preload_fn, dataset_transform, dataset_transform_train):
    small_hooks_mf0 = [MetricsHook_mf0(), VisualOutputHook_mf0(True, True, True)]
    small_hooks = [MetricsHook(), VisualOutputHook(True, True, False)]
    valid_hooks = [MetricsHook(), VisualOutputHook(False, False, True), SaveBestModelHook()]

    validation_datasets = []
    train_data = []
    if datasets.wjazzd.prefix in which:
        wjazzd_train, wjazzd_validation, wjazzd_small_validation = datasets.wjazzd.prepare(preload_fn)
        wjazzd_validation_dataset = datasets.AADataset(wjazzd_validation, args, dataset_transform)
        wjazzd_small_validation_dataset = datasets.AADataset(wjazzd_small_validation, args, dataset_transform)
        validation_datasets += [
            VD("small_"+datasets.wjazzd.prefix, wjazzd_small_validation_dataset, 3000, small_hooks),
            VD(datasets.wjazzd.prefix, wjazzd_validation_dataset, 30000, valid_hooks),
        ]
        train_data += wjazzd_train

    if datasets.medleydb.prefix in which:
        medleydb_train, medleydb_validation, medleydb_small_validation = datasets.medleydb.prepare(preload_fn)
        medleydb_validation_dataset = datasets.AADataset(medleydb_validation, args, dataset_transform)
        medleydb_small_validation_dataset = datasets.AADataset(medleydb_small_validation, args, dataset_transform)
        validation_datasets += [
            VD("small_"+datasets.medleydb.prefix, medleydb_small_validation_dataset, 3000, small_hooks),
            VD(datasets.medleydb.prefix, medleydb_validation_dataset, 20000, valid_hooks),
        ]
        train_data += medleydb_train

    if datasets.mdb_melody_synth.prefix in which:
        mdb_melody_synth_train, mdb_melody_synth_validation, _ = datasets.mdb_melody_synth.prepare(preload_fn)
        mdb_melody_synth_validation_dataset = datasets.AADataset(mdb_melody_synth_validation, args, dataset_transform)
        validation_datasets += [
            VD(datasets.mdb_melody_synth.prefix, mdb_melody_synth_validation_dataset, 30000, valid_hooks),
        ]
        train_data += mdb_melody_synth_train

    if datasets.mdb_stem_synth.prefix in which:
        mdb_stem_synth_train, mdb_stem_synth_validation, mdb_stem_synth_small_validation = datasets.mdb_stem_synth.prepare(preload_fn)
        mdb_stem_synth_small_validation_dataset = datasets.AADataset(mdb_stem_synth_small_validation, args, dataset_transform)
        mdb_stem_synth_validation_dataset = datasets.AADataset(mdb_stem_synth_validation, args, dataset_transform)
        validation_datasets += [
            VD("small_"+datasets.mdb_stem_synth.prefix, mdb_stem_synth_small_validation_dataset, 3000, small_hooks),
            VD(datasets.mdb_stem_synth.prefix, mdb_stem_synth_validation_dataset, 3000, valid_hooks),
        ]
        train_data += mdb_stem_synth_train

    if datasets.mdb_mf0_synth.prefix in which:
        _, _, mdb_mf0_synth_small_validation = datasets.mdb_mf0_synth.prepare(preload_fn)
        mdb_mf0_synth_small_validation_dataset = datasets.AADataset(mdb_mf0_synth_small_validation, args, dataset_transform)
        validation_datasets += [
            VD("small_"+datasets.mdb_mf0_synth.prefix, mdb_mf0_synth_small_validation_dataset, 3000, small_hooks_mf0),
        ]

    train_dataset = datasets.AADataset(train_data, args, dataset_transform_train, shuffle=True)

    return train_dataset, validation_datasets
