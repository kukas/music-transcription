import tensorflow as tf
import json
import datasets
import datetime
import time
from model import VD, VisualOutputHook, MetricsHook, MetricsHook_mf0, VisualOutputHook_mf0, SaveBestModelHook


def name(args, prefix=""):
    if args.logdir is None:
        filtered = ["threads", "logdir", "epochs", "debug_memory_leaks", "full_trace", "evaluate", "note_range", "checkpoint"]
        name = "{}-{}".format(datetime.datetime.now().strftime("%m%d_%H%M%S"), prefix)
        for k, v in vars(args).items():
            if k not in filtered:
                short_k = "".join([w[0] for w in k.split("_")])
                if type(v) is list or type(v) is tuple:
                    v = ",".join(v)
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
    timer = time.time()

    small_hooks_mf0 = [MetricsHook_mf0(), VisualOutputHook_mf0(True, True, True)]
    small_hooks = [MetricsHook(), VisualOutputHook(True, True, False, False)]
    valid_hooks = [MetricsHook(), VisualOutputHook(False, False, True, True), SaveBestModelHook(args.logdir)]
    test_hooks = [MetricsHook(write_summaries=False, print_detailed=True, write_estimations=True)]

    validation_datasets = []
    test_datasets = []
    train_data = []
    if datasets.medleydb.prefix in which:
        medleydb_train, medleydb_test, medleydb_validation, medleydb_small_validation = datasets.medleydb.prepare(preload_fn)
        medleydb_test_dataset = datasets.AADataset(medleydb_test, args, dataset_transform)
        medleydb_validation_dataset = datasets.AADataset(medleydb_validation, args, dataset_transform)
        medleydb_small_validation_dataset = datasets.AADataset(medleydb_small_validation, args, dataset_transform)
        validation_datasets += [
            VD("small_"+datasets.medleydb.prefix, medleydb_small_validation_dataset, 5000, small_hooks),
            VD(datasets.medleydb.prefix, medleydb_validation_dataset, 20000, valid_hooks),
        ]
        test_datasets += [
            VD(datasets.medleydb.prefix, medleydb_test_dataset, 0, test_hooks),
            VD(datasets.medleydb.prefix, medleydb_validation_dataset, 0, test_hooks),
        ]
        train_data += medleydb_train

    if datasets.mdb_melody_synth.prefix in which:
        mdb_melody_synth_train, mdb_melody_synth_test, mdb_melody_synth_validation, _ = datasets.mdb_melody_synth.prepare(preload_fn)
        mdb_melody_synth_test_dataset = datasets.AADataset(mdb_melody_synth_test, args, dataset_transform)
        mdb_melody_synth_validation_dataset = datasets.AADataset(mdb_melody_synth_validation, args, dataset_transform)
        validation_datasets += [
            VD(datasets.mdb_melody_synth.prefix, mdb_melody_synth_validation_dataset, 40000, valid_hooks),
        ]
        test_datasets += [
            VD(datasets.mdb_melody_synth.prefix, mdb_melody_synth_test_dataset, 0, test_hooks),
            VD(datasets.mdb_melody_synth.prefix, mdb_melody_synth_validation_dataset, 0, test_hooks),
        ]
        train_data += mdb_melody_synth_train

    if datasets.mdb_stem_synth.prefix in which:
        mdb_stem_synth_train, mdb_stem_synth_validation, mdb_stem_synth_small_validation = datasets.mdb_stem_synth.prepare(preload_fn)
        mdb_stem_synth_small_validation_dataset = datasets.AADataset(mdb_stem_synth_small_validation, args, dataset_transform)
        mdb_stem_synth_validation_dataset = datasets.AADataset(mdb_stem_synth_validation, args, dataset_transform)
        validation_datasets += [
            VD("small_"+datasets.mdb_stem_synth.prefix, mdb_stem_synth_small_validation_dataset, 5000, small_hooks),
            VD(datasets.mdb_stem_synth.prefix, mdb_stem_synth_validation_dataset, 40000, valid_hooks),
        ]
        train_data += mdb_stem_synth_train

    if datasets.mdb_mf0_synth.prefix in which:
        _, _, mdb_mf0_synth_small_validation = datasets.mdb_mf0_synth.prepare(preload_fn)
        mdb_mf0_synth_small_validation_dataset = datasets.AADataset(mdb_mf0_synth_small_validation, args, dataset_transform)
        validation_datasets += [
            VD("small_"+datasets.mdb_mf0_synth.prefix, mdb_mf0_synth_small_validation_dataset, 5000, small_hooks_mf0),
        ]
    
    if datasets.wjazzd.prefix in which:
        wjazzd_train, wjazzd_test, wjazzd_validation, wjazzd_small_validation = datasets.wjazzd.prepare(preload_fn)
        wjazzd_test_dataset = datasets.AADataset(wjazzd_test, args, dataset_transform)
        wjazzd_validation_dataset = datasets.AADataset(wjazzd_validation, args, dataset_transform)
        wjazzd_small_validation_dataset = datasets.AADataset(wjazzd_small_validation, args, dataset_transform)
        validation_datasets += [
            VD("small_"+datasets.wjazzd.prefix, wjazzd_small_validation_dataset, 5000, small_hooks),
            VD(datasets.wjazzd.prefix, wjazzd_validation_dataset, 40000, valid_hooks),
        ]
        test_datasets += [
            VD(datasets.wjazzd.prefix, wjazzd_test_dataset, 0, test_hooks),
            VD(datasets.wjazzd.prefix, wjazzd_validation_dataset, 0, test_hooks),
        ]
        train_data += wjazzd_train

    if datasets.orchset.prefix in which:
        orchset_test, orchset_small_validation = datasets.orchset.prepare(preload_fn)
        orchset_test_dataset = datasets.AADataset(orchset_test, args, dataset_transform)
        orchset_small_validation_dataset = datasets.AADataset(orchset_small_validation, args, dataset_transform)
        validation_datasets.append(VD("small_"+datasets.orchset.prefix, orchset_small_validation_dataset, 5000, small_hooks))
        test_datasets.append(VD(datasets.orchset.prefix, orchset_test_dataset, 0, test_hooks))

    if datasets.adc2004.prefix in which:
        adc2004_test = datasets.adc2004.prepare(preload_fn)
        adc2004_test_dataset = datasets.AADataset(adc2004_test, args, dataset_transform)
        test_datasets.append(VD(datasets.adc2004.prefix, adc2004_test_dataset, 0, test_hooks))

    if datasets.mirex05.prefix in which:
        mirex05_test = datasets.mirex05.prepare(preload_fn)
        mirex05_test_dataset = datasets.AADataset(mirex05_test, args, dataset_transform)
        test_datasets.append(VD(datasets.mirex05.prefix, mirex05_test_dataset, 0, test_hooks))

    if train_data:
        train_dataset = datasets.AADataset(train_data, args, dataset_transform_train, shuffle=True)
    else:
        # Return at least one dataset as training, since its parameters are used in network initialization
        train_dataset = (test_datasets+validation_datasets)[0].dataset

    print("datasets ready in {:.2f}s".format(time.time() - timer))

    return train_dataset, test_datasets, validation_datasets
