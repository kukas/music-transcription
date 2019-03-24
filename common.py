import argparse
import tensorflow as tf
import json
import datasets
import datetime
import time
import sys
from model import VD, VisualOutputHook, MetricsHook, MetricsHook_mf0, VisualOutputHook_mf0, SaveBestModelHook

# Console argument functions

def name(args, prefix=""):
    if args.logdir is None:
        filtered = ["logdir", "checkpoint", "saver_max_to_keep", "threads", "full_trace", "debug_memory_leaks", "cpu" ,"rewind", "evaluate", "epochs", "batch_size_evaluation", "note_range"]
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

def common_arguments(defaults={}):
    _defaults = {
        "annotations_per_window": 1,
        "hop_size": None,
        "frame_width": round(256/(44100/16000)),
        "context_width": 0,
        "samplerate": 16000,
        "min_note": 0,
        "note_range": 128,
    }
    defaults = {**_defaults, **defaults}
    parser = argparse.ArgumentParser()
    # loading models
    parser.add_argument("--logdir", default=None, type=str, help="Path to model directory.")
    parser.add_argument("--checkpoint", default="model", type=str, help="Checkpoint name.")
    parser.add_argument("--saver_max_to_keep", default=1, type=int, help="How many checkpoints to keep")
    # debug & system settings
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--full_trace", action='store_true', help="Profile Tensorflow session.")
    parser.add_argument("--debug_memory_leaks", action='store_true', help="Debug memory leaks.")
    parser.add_argument("--cpu", action='store_true', help="Disable GPU.")
    # training settings
    parser.add_argument("--rewind", action='store_true', help="Rewind back to the same point in training.")
    parser.add_argument("--evaluate", action='store_true', help="Evaluate after training. If an existing checkpoint is specified, it will be evaluated only.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train for.")
    # input
    parser.add_argument("--datasets", default=["mdb"], nargs="+", type=str, help="Datasets to use for this experiment")
    parser.add_argument("--batch_size", default=32, type=int, help="Number of examples in one batch")
    parser.add_argument("--batch_size_evaluation", default=64, type=int, help="Number of examples in one batch for evaluation")
    # input window shape
    parser.add_argument("--samplerate", default=defaults["samplerate"], type=int, help="Audio samplerate used in the model, resampling is done automatically.")
    parser.add_argument("--hop_size", default=defaults["hop_size"], type=int, help="Hop of the input window specified in number of annotations. Defaults to annotations_per_window")
    parser.add_argument("--frame_width", default=defaults["frame_width"], type=int, help="Number of samples per annotation = hop size.")
    parser.add_argument("--context_width", default=defaults["context_width"], type=int, help="Number of context samples on both sides of the example window.")
    parser.add_argument("--input_normalization", action='store_true', default=True, help="Enable normalizing each input example")
    parser.add_argument("--no_input_normalization", action='store_true', dest='input_normalization', help="Disable normalizing each input example")
    # output notes shape
    parser.add_argument("--annotations_per_window", default=defaults["annotations_per_window"], type=int, help="Number of annotations in one example.")
    parser.add_argument("--min_note", default=defaults["min_note"], type=int, help="First MIDI note number.")
    parser.add_argument("--note_range", default=defaults["note_range"], type=int, help="Note range.")
    parser.add_argument("--bins_per_semitone", default=5, type=int, help="Bins per semitone")
    parser.add_argument("--annotation_smoothing", default=0.25, type=float, help="Set standard deviation of the gaussian blur for the frame annotations")

    # learning parameters
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--clip_gradients", default=0.0, type=float, help="Clip gradients by global norm")
    parser.add_argument("--l2_loss_weight", default=0.0, type=float, help="L2 loss weight")
    parser.add_argument("--miss_weight", default=1.0, type=float, help="Weight for missed frames in the loss function")

    return parser

# Model functions

def input_normalization(window, args):
    if args.input_normalization:
        mean, var = tf.nn.moments(window, axes=[1])
        mean = tf.expand_dims(mean, axis=1)

        epsilon = 1e-3
        std_inv = tf.math.rsqrt(var + epsilon)
        std_inv = tf.expand_dims(std_inv, axis=1)

        return (window - mean) * std_inv

def loss(self, args):
    # Melody input, not compatible with multif0 input
    annotations = self.annotations[:, :, 0] - args.min_note
    voicing_ref = tf.cast(tf.greater(annotations, 0), tf.float32)
    if args.annotation_smoothing > 0:
        peak_ref = tf.cast(tf.abs(tf.tile(tf.reshape(annotations, [-1, self.annotations_per_window, 1]), [1, 1, self.bin_count]) - self.note_bins) < 0.5, tf.float32)
        self.note_probabilities = tf.nn.sigmoid(self.note_logits)
        note_ref = tf.tile(tf.reshape(annotations, [-1, self.annotations_per_window, 1]), [1, 1, self.bin_count])
        ref_probabilities = tf.exp(-(note_ref-self.note_bins)**2/(args.annotation_smoothing**2))

        voicing_weights = tf.tile(tf.expand_dims(voicing_ref, -1), [1, 1, self.bin_count])
        miss_weights = tf.ones_like(voicing_weights)*args.miss_weight + peak_ref*(1-args.miss_weight)
        loss = tf.losses.sigmoid_cross_entropy(ref_probabilities, self.note_logits, weights=voicing_weights*miss_weights)
    else:
        self.note_probabilities = tf.nn.softmax(self.note_logits)
        ref_bins = tf.cast(tf.round(annotations * self.bins_per_semitone), tf.int32)
        loss = tf.losses.sparse_softmax_cross_entropy(ref_bins, self.note_logits, weights=voicing_ref)
    
    if args.l2_loss_weight > 0:
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_loss = tf.reduce_sum(tf.constant(args.l2_loss_weight)*reg_variables)
        tf.summary.scalar('metrics/train/l2_loss', l2_loss)
        tf.summary.scalar('metrics/train/ce_loss', loss)
        loss += l2_loss
    
    return loss

def est_notes(self, args):
    peak_est = tf.cast(tf.argmax(self.note_logits, axis=2) / self.bins_per_semitone, tf.float32)
    peak_est = tf.cast(tf.abs(tf.tile(tf.reshape(peak_est, [-1, self.annotations_per_window, 1]), [1, 1, self.bin_count]) - self.note_bins) < 0.5, tf.float32)
    probs_around_peak = self.note_probabilities*peak_est
    probs_around_peak_sums = tf.reduce_sum(probs_around_peak, axis=2)

    est_notes = (tf.reduce_sum(self.note_bins * probs_around_peak, axis=2)/probs_around_peak_sums + args.min_note)

    if self.voicing_logits is not None:
        est_notes *= tf.cast(tf.greater(self.voicing_logits, 0), tf.float32)*2 - 1
    else:
        est_notes *= tf.cast(tf.greater(probs_around_peak_sums, 0.3), tf.float32)*2 - 1

    return est_notes

def optimizer(self, args):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        # Get the gradient pairs (Tensor, Variable)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        if args.clip_gradients > 0:
            grads, tvars = zip(*grads_and_vars)
            grads, _ = tf.clip_by_global_norm(grads, args.clip_gradients)
            grads_and_vars = list(zip(grads, tvars))
        # Update the weights wrt to the gradient
        training = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # Save the grads and vars with tf.summary.histogram
        for grad, var in grads_and_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
            tf.summary.histogram(var.name, var)

    return training

def bn_conv(inputs, filters, size, strides, padding, activation=None, dilation_rate=1, training=False, reuse=None):
    name = "bn_conv{}-f{}-s{}-dil{}-{}".format(size, filters, strides, dilation_rate, padding)
    with tf.name_scope(name):
        l = tf.layers.conv1d(inputs, filters, size, strides, padding, activation=None, use_bias=False, dilation_rate=dilation_rate, reuse=reuse)
        l = tf.layers.batch_normalization(l, training=training, reuse=reuse)
        if activation:
            return activation(l)
        else:
            return l


def conv(inputs, filters, size, strides, padding, activation=None, dilation_rate=1, training=False):
    name = "conv{}-f{}-s{}-dil{}-{}".format(size, filters, strides, dilation_rate, padding)
    return tf.layers.conv1d(inputs, filters, size, strides, padding, activation=activation, dilation_rate=dilation_rate, name=name)


def add_layers_from_string(in_layer, layers_string):
    print("constructing", layers_string)
    for layer in layers_string.split("->"):
        params = layer.split("_")
        layer_type = params.pop(0)
        p = {x[0]: x[1:] for x in params}
        if layer_type == "conv":
            activation = None
            if "a" in p:
                if p["a"] == "relu":
                    activation = tf.nn.relu
                if p["a"] == "tanh":
                    activation = tf.nn.tanh
            in_layer = tf.layers.conv1d(in_layer, filters=int(p["f"]), kernel_size=int(p["k"]), 
                                        strides=int(p["s"]), padding=p["P"], activation=activation)

        if layer_type == "avgpool":
            in_layer = tf.layers.average_pooling1d(in_layer, pool_size=int(p["p"]), strides=int(p["s"]), padding=p["P"])
        
        print(in_layer)

    return in_layer
# Dataset functions

def prepare_datasets(which, args, preload_fn, dataset_transform, dataset_transform_train):
    timer = time.time()

    small_hooks_mf0 = [MetricsHook_mf0(), VisualOutputHook_mf0(True, True, True)]
    small_hooks = [MetricsHook(), VisualOutputHook(True, True, False, False)]
    valid_hooks = [MetricsHook(write_estimations=True), VisualOutputHook(False, False, True, True), SaveBestModelHook(args.logdir)]
    test_hooks = [MetricsHook(write_summaries=False, print_detailed=True, write_estimations=True)]

    validation_datasets = []
    test_datasets = []
    train_data = []
    if datasets.medleydb.prefix in which:
        medleydb_train, medleydb_test, medleydb_validation, medleydb_small_validation = datasets.medleydb.prepare(preload_fn, threads=args.threads)
        medleydb_test_dataset = datasets.AADataset(medleydb_test, args, dataset_transform)
        medleydb_validation_dataset = datasets.AADataset(medleydb_validation, args, dataset_transform)
        medleydb_small_validation_dataset = datasets.AADataset(medleydb_small_validation, args, dataset_transform)
        validation_datasets += [
            VD("small_"+datasets.medleydb.prefix, medleydb_small_validation_dataset, 1000, small_hooks),
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

    if "fairerhopes" in which:
        harfa_audio = datasets.Audio("/mnt/tera/jirka/V1/MatthewEntwistle_FairerHopes/MatthewEntwistle_FairerHopes_STEMS/MatthewEntwistle_FairerHopes_STEM_10.wav",
                    "augment_low")
        harfa_annot = datasets.Annotation.from_time_series("data/MatthewEntwistle_FairerHopes_STEM_10_clean.csv", "fairerhopes")
        harfa = datasets.AnnotatedAudio(harfa_annot, harfa_audio)
        preload_fn(harfa)
        # harfa.audio.samples *= 5
        mirex05_test_dataset = datasets.AADataset([harfa], args, dataset_transform)
        test_datasets.append(VD("fairerhopes", mirex05_test_dataset, 0, test_hooks))

    if train_data:
        hop_size = args.hop_size if args.hop_size is not None else None
        train_dataset = datasets.AADataset(train_data, args, dataset_transform_train, shuffle=True, hop_size=hop_size)
    else:
        # Return at least one dataset as training, since its parameters are used in network initialization
        train_dataset = (test_datasets+validation_datasets)[0].dataset

    print("datasets ready in {:.2f}s".format(time.time() - timer))

    return train_dataset, test_datasets, validation_datasets

# Main functions


def main(argv, construct, parse_args):
    args = parse_args(argv)
    # Construct the network
    network, train_dataset, validation_datasets, test_datasets = construct(args)

    if not (args.evaluate and network.restored):
        try:
            network.train(train_dataset, args.epochs, validation_datasets, save_every_n_batches=10000)
            network.save(args.checkpoint)
        except KeyboardInterrupt:
            network.save(args.checkpoint)
            sys.exit()

    if args.evaluate:
        for vd in test_datasets:
            print("{} evaluation".format(vd.name))
            network.evaluate(vd)
