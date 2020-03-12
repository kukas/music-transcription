import argparse
import tensorflow as tf
import json
import datasets
import evaluation
import datetime
import time
import sys
from model import VD, VisualOutputHook, MetricsHook, MetricsHook_mf0, VisualOutputHook_mf0, SaveBestModelHook, safe_div, SaveSaliencesHook, CSVOutputWriterHook
import librosa
import numpy as np
import os
# Console argument functions

def name(args, specified_args=[], prefix=""):
    if args.logdir is None:
        filtered = ["logdir", "checkpoint", "saver_max_to_keep", "threads", "full_trace", "debug_memory_leaks", "cpu",
                    "rewind", "save_salience", "predict", "output_file", "output_format", "evaluate", "evaluate_every", "evaluate_small_every", "epochs", "iterations", "batch_size_evaluation", "stop_if_too_slow"]
        name = "{}-{}".format(datetime.datetime.now().strftime("%m%d_%H%M%S"), prefix)
        print(specified_args)
        for k, v in vars(args).items():
            if k not in filtered and ((specified_args != [] and k in specified_args) or specified_args == []):
                short_k = "".join([w[0] for w in k.split("_")])
                if type(v) is list or type(v) is tuple:
                    v = map(str, v)
                    v = ",".join(v)
                name += "-{}{}".format(short_k, v)

        args.logdir = "models/" + name

        print()
        print(name)
        print()

def common_arguments_parser():
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
    parser.add_argument("--stop_if_too_slow", default=None, type=float, help="Exit training if training one batch takes longer than N seconds.")
    # training settings
    parser.add_argument("--rewind", action='store_true', help="Rewind back to the same point in training.")
    parser.add_argument("--save_salience", action='store_true', help="Save salience output when evaluating.")
    parser.add_argument("--predict", default=None, type=str, help="Extract the melody from the input file.")
    parser.add_argument("--output_file", default=None, type=str, help="Extract the melody to the output file.")
    parser.add_argument("--output_format", default="mirex", type=str, help="output file format.")
    parser.add_argument("--evaluate", action='store_true', help="Evaluate after training. If an existing checkpoint is specified, it will be evaluated only.")
    parser.add_argument("--evaluate_every", type=int, help="Evaluate validation set every N steps.")
    parser.add_argument("--evaluate_small_every", type=int, help="Evaluate small validation set every N steps.")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs to train for.")
    parser.add_argument("--iterations", default=None, type=int, help="Number of iterations to train for.")
    # input
    parser.add_argument("--datasets", nargs="+", type=str, help="Datasets to use for this experiment")
    parser.add_argument("--batch_size", type=int, help="Number of examples in one batch")
    parser.add_argument("--batch_size_evaluation", default=64, type=int, help="Number of examples in one batch for evaluation")
    # input window shape
    parser.add_argument("--samplerate", type=int, help="Audio samplerate used in the model, resampling is done automatically.")
    parser.add_argument("--hop_size", type=int, help="Hop of the input window specified in number of annotations. Defaults to annotations_per_window")
    parser.add_argument("--frame_width", type=int, help="Number of samples per annotation = hop size. !!!!!!! POZOR, PŘI RESAMPLINGU VALIDAČNÍCH DAT PAK NESEDÍ FINÁLNÍ VÝSLEDKY S TĚMI Z TENSORBOARDU !!!!!!!!")
    parser.add_argument("--context_width", type=int, help="Number of context samples on both sides of the example window.")
    # output notes shape
    parser.add_argument("--annotations_per_window", type=int, help="Number of annotations in one example.")
    parser.add_argument("--min_note", type=int, help="First MIDI note number.")
    parser.add_argument("--note_range", type=int, help="Note range.")
    parser.add_argument("--bins_per_semitone", type=int, help="Bins per semitone")
    parser.add_argument("--annotation_smoothing", type=float, help="Set standard deviation of the gaussian blur for the frame annotations")
    parser.add_argument("--peak_est_averaging", type=float, help="Width of the window for local note peak averaging")
    # learning parameters
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--learning_rate_decay", type=float, help="Learning rate decay")
    parser.add_argument("--learning_rate_decay_steps", type=int, help="Learning rate decay steps")
    parser.add_argument("--clip_gradients", type=float, help="Clip gradients by global norm")
    parser.add_argument("--unvoiced_loss_weight", type=float, help="Unvoiced frames loss weight for the melody model")
    parser.add_argument("--l2_loss_weight", type=float, help="L2 loss weight")
    parser.add_argument("--miss_weight", type=float, help="Weight for missed frames in the loss function")

    return parser

def argument_defaults(args, defaults):
    arg_defaults = {
        "batch_size": 32,
        "annotations_per_window": 1,
        "hop_size": None,
        "frame_width": round(256/(44100/16000)),
        "context_width": 0,
        "samplerate": 16000,
        "min_note": 0,
        "note_range": 128,
        "evaluate_every": 20000,
        "evaluate_small_every": 5000,
        "bins_per_semitone": 5,
        "annotation_smoothing": 0.25,
        "peak_est_averaging": 0.25,
        "datasets": ["mdb"],
        "learning_rate": 0.001,
        "learning_rate_decay": 1.0,
        "learning_rate_decay_steps": 100000,
        "clip_gradients": 0.0,
        "unvoiced_loss_weight": 0.0,
        "l2_loss_weight": 0.0,
        "miss_weight": 1.0,
    }

    defaults = {**arg_defaults, **defaults}

    specified_args = []
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
        else:
            specified_args.append(key)

    return specified_args

# Model functions

def input_normalization(window, args):
    if args.input_normalization:
        mean, var = tf.nn.moments(window, axes=[1])
        mean = tf.expand_dims(mean, axis=1)

        epsilon = 1e-3
        std_inv = tf.math.rsqrt(var + epsilon)
        std_inv = tf.expand_dims(std_inv, axis=1)

        return (window - mean) * std_inv

def _common_losses(self, args):
    loss_names = []
    losses = []
    if args.l2_loss_weight > 0:
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_loss = tf.reduce_sum(tf.constant(args.l2_loss_weight)*reg_variables)

        loss_names.append("l2_loss")
        losses.append(l2_loss)

    return loss_names, losses

def _common_loss_metrics(self, loss_names, losses):
    if len(losses) > 1:
        for name, loss in zip(loss_names, losses):
            tf.summary.scalar('metrics/train/'+name, loss)

def loss(self, args):
    with tf.name_scope("losses"):
        # Melody input, not compatible with multif0 input
        annotations = self.annotations[:, :, 0] - args.min_note
        voicing_ref = tf.cast(tf.greater(annotations, 0), tf.float32)
        loss_names = []
        losses = []
        if self.note_logits is not None:
            if args.annotation_smoothing > 0:
                self.note_probabilities = tf.nn.sigmoid(self.note_logits)
                note_ref = tf.tile(tf.reshape(annotations, [-1, self.annotations_per_window, 1]), [1, 1, self.bin_count])
                ref_probabilities = tf.exp(-(note_ref-self.note_bins)**2/(2*args.annotation_smoothing**2))

                unvoiced_weights = (1-voicing_ref)*args.unvoiced_loss_weight
                voicing_weights = tf.tile(tf.expand_dims(voicing_ref+unvoiced_weights, -1), [1, 1, self.bin_count])

                # miss weights
                peak_ref = tf.cast(tf.abs(tf.tile(tf.reshape(annotations, [-1, self.annotations_per_window, 1]), [1, 1, self.bin_count]) - self.note_bins) < 0.5, tf.float32)
                miss_weights = tf.ones_like(voicing_weights)*args.miss_weight + peak_ref*(1-args.miss_weight)

                note_loss = tf.losses.sigmoid_cross_entropy(ref_probabilities, self.note_logits, weights=voicing_weights*miss_weights)
            else:
                self.note_probabilities = tf.nn.softmax(self.note_logits)
                ref_bins = tf.cast(tf.round(annotations * self.bins_per_semitone), tf.int32)
                note_loss = tf.losses.sparse_softmax_cross_entropy(ref_bins, self.note_logits, weights=voicing_ref)

            loss_names.append("note_loss")
            losses.append(note_loss)
        
        if self.voicing_logits is not None:
            voicing_loss = tf.losses.sigmoid_cross_entropy(voicing_ref, self.voicing_logits)

            loss_names.append("voicing_loss")
            losses.append(voicing_loss)
    
        add_loss_names, add_losses = _common_losses(self, args)
        loss_names += add_loss_names
        losses += add_losses

    _common_loss_metrics(self, loss_names, losses)

    return tf.math.add_n(losses)

def loss_mf0(self, args):
    # multif0 loss ---------
    with tf.name_scope("losses"):
        annotations = self.annotations - args.min_note

        loss_names = []
        losses = []
        if self.note_logits is not None:
            self.note_probabilities = tf.nn.sigmoid(self.note_logits)
            if args.annotation_smoothing > 0:
                print("self.note_probabilities.shape", self.note_probabilities.shape)
                annotations_per_frame = tf.shape(annotations)[-1]
                note_bins = tf.tile(tf.expand_dims(self.note_bins, 2), [1, 1, annotations_per_frame, 1])
                print("note_bins.shape", note_bins.shape)
                note_ref = tf.tile(tf.reshape(annotations, [-1, self.annotations_per_window, annotations_per_frame, 1]), [1, 1, 1, self.bin_count])
                ref_probabilities = tf.exp(-(note_ref-note_bins)**2/(2*args.annotation_smoothing**2))
                ref_probabilities = tf.reduce_sum(ref_probabilities, axis=2)
            else:
                ref_probabilities = tf.reduce_sum(tf.one_hot(tf.cast(annotations, tf.int32), self.note_range), axis=2)
                ref_probabilities = tf.cast(tf.greater(ref_probabilities, 0), tf.float32)

            note_loss = tf.losses.sigmoid_cross_entropy(ref_probabilities, self.note_logits)  # , weights=voicing_weights)

            tf.summary.image("training/ref_probabilities", tf.expand_dims(ref_probabilities, -1), max_outputs=1)
            tf.summary.image("training/note_logits", tf.expand_dims(self.note_logits, -1), max_outputs=1)

            loss_names.append("note_loss")
            losses.append(note_loss)

        add_loss_names, add_losses = _common_losses(self, args)
        loss_names += add_loss_names
        losses += add_losses

    _common_loss_metrics(self, loss_names, losses)

    return tf.math.add_n(losses)

def loss_mir(self, args):
    # multi-instrument recognition
    with tf.name_scope("losses"):
        annotations = self.annotations

        loss_names = []
        losses = []
        if self.note_logits is not None:
            self.note_probabilities = tf.nn.sigmoid(self.note_logits)
            ref_probabilities = tf.reduce_sum(tf.one_hot(tf.cast(annotations, tf.int32), self.note_range), axis=2)
            ref_probabilities = tf.cast(tf.greater(ref_probabilities, 0), tf.float32)
            print("ref_probabilities.shape", ref_probabilities.shape)

            note_loss = tf.losses.sigmoid_cross_entropy(ref_probabilities, self.note_logits)

            tf.summary.image("training/ref_probabilities", tf.expand_dims(ref_probabilities, -1), max_outputs=1)
            tf.summary.image("training/note_logits", tf.expand_dims(self.note_logits, -1), max_outputs=1)

            loss_names.append("note_loss")
            losses.append(note_loss)

        add_loss_names, add_losses = _common_losses(self, args)
        loss_names += add_loss_names
        losses += add_losses

    _common_loss_metrics(self, loss_names, losses)

    return tf.math.add_n(losses)

def est_notes(self, args):
    with tf.name_scope("est_notes"):
        peak_est = tf.cast(tf.argmax(self.note_probabilities, axis=2) / self.bins_per_semitone, tf.float32)
        if args.peak_est_averaging > 0.0:
            peak_est_mask = tf.cast(tf.abs(tf.tile(tf.reshape(peak_est, [-1, self.annotations_per_window, 1]), [1, 1, self.bin_count]) - self.note_bins) < args.peak_est_averaging, tf.float32)
            probs_around_peak = self.note_probabilities*peak_est_mask
            probs_around_peak_sums = tf.reduce_sum(probs_around_peak, axis=2)
            # self.est_notes_confidence = probs_around_peak_sums / tf.math.count_nonzero(peak_est_mask, axis=2, dtype=tf.float32)

            est_notes = safe_div(tf.reduce_sum(self.note_bins * probs_around_peak, axis=2), probs_around_peak_sums) + args.min_note
        else:
            est_notes = peak_est + args.min_note
        
        self.est_notes_confidence = tf.reduce_max(self.note_probabilities, axis=2)

        if self.voicing_logits is not None:
            est_notes *= tf.cast(tf.greater(self.voicing_logits, 0), tf.float32)*2 - 1
        else:
            est_notes *= tf.cast(tf.greater(self.est_notes_confidence, self.voicing_threshold), tf.float32)*2 - 1

    return est_notes


def optimizer(self, args):
    with tf.name_scope("optimizer"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            learning_rate = args.learning_rate
            if args.learning_rate_decay < 1.0:
                learning_rate = tf.train.exponential_decay(args.learning_rate, self.global_step, args.learning_rate_decay_steps, args.learning_rate_decay, True)
                tf.summary.scalar("metrics/train/learning_rate", learning_rate)

            optimizer = tf.train.AdamOptimizer(learning_rate)
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

def spectrograms(args):
    HOP_LENGTH = args.frame_width
    if args.spectrogram == "hcqt":
        HARMONICS = [0.5, 1, 2, 3, 4, 5]
        FMIN = 32.7
        BINS_PER_OCTAVE = 60
        N_BINS = BINS_PER_OCTAVE*6

        def spectrogram_function(audio, samplerate):
            cqt_list = []
            shapes = []
            for h in HARMONICS:
                cqt = librosa.cqt(
                    audio, sr=samplerate, hop_length=HOP_LENGTH, fmin=FMIN*float(h),
                    n_bins=N_BINS,
                    bins_per_octave=BINS_PER_OCTAVE
                )
                cqt_list.append(cqt)
                shapes.append(cqt.shape)

            shapes_equal = [s == shapes[0] for s in shapes]
            if not all(shapes_equal):
                print("NOT ALL", shapes_equal)
                min_time = np.min([s[1] for s in shapes])
                new_cqt_list = []
                for i in range(len(cqt_list)):
                    new_cqt_list.append(cqt_list[i][:, :min_time])
                cqt_list = new_cqt_list

            log_hcqt = ((1.0/80.0) * librosa.core.amplitude_to_db(
                np.abs(np.array(cqt_list)), ref=np.max)) + 1.0
            
            return (log_hcqt*65535).astype(np.uint16)

        spectrogram_thumb = "hcqt-fmin{}-oct{}-octbins{}-hop{}-db-uint16".format(FMIN, N_BINS/BINS_PER_OCTAVE, BINS_PER_OCTAVE, HOP_LENGTH)
        spectrogram_info = (len(HARMONICS), N_BINS, HOP_LENGTH, FMIN)

    elif args.spectrogram == "cqt":
        FMIN = 32.7
        BINS_PER_OCTAVE = 60
        N_BINS = BINS_PER_OCTAVE*9
        top_db = args.spectrogram_top_db
        filter_scale = args.spectrogram_filter_scale

        def spectrogram_function(audio, samplerate):
            cqt = librosa.cqt(audio, sr=samplerate, hop_length=HOP_LENGTH, fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, filter_scale=filter_scale)

            log_cqt = (librosa.core.amplitude_to_db(np.abs(cqt), ref=np.max, top_db=top_db) / top_db) + 1.0
            log_cqt = np.expand_dims(log_cqt, 0)
            return (log_cqt*65535).astype(np.uint16)

        spectrogram_thumb = "cqt-fmin{}-oct{}-octbins{}-hop{}-db{}-fs{}-uint16".format(FMIN, N_BINS/BINS_PER_OCTAVE, BINS_PER_OCTAVE, HOP_LENGTH, top_db, filter_scale)
        spectrogram_info = (1, N_BINS, HOP_LENGTH, FMIN)

    elif args.spectrogram == "cqt_fs":
        filter_scales = [0.5, 1, 2]
        top_db = [60, 80, 100]
        FMIN = 32.7
        BINS_PER_OCTAVE = 60
        N_BINS = BINS_PER_OCTAVE*9

        def spectrogram_function(audio, samplerate):
            cqts = []
            for fscale, db in zip(filter_scales, top_db):
                cqt = librosa.cqt(audio, sr=samplerate, hop_length=HOP_LENGTH, fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, filter_scale=fscale)
                log_cqt = (librosa.core.amplitude_to_db(np.abs(cqt), ref=np.max, top_db=db) / db) + 1.0
                cqts.append(log_cqt)
            return (np.array(cqts)*65535).astype(np.uint16)

        spectrogram_thumb = "cqt-fmin{}-oct{}-octbins{}-hop{}-db-fs{}-uint16".format(FMIN, N_BINS/BINS_PER_OCTAVE, BINS_PER_OCTAVE, HOP_LENGTH, "+".join(map(str, filter_scales)))
        spectrogram_info = (len(filter_scales), N_BINS, HOP_LENGTH, FMIN)

    return spectrogram_function, spectrogram_thumb, spectrogram_info

def harmonic_stacking(self, input, undertones, overtones, bin_count=None, bins_per_semitone=None, offset=0):
    if bin_count is None:
        bin_count = self.bin_count
    if bins_per_semitone is None:
        bins_per_semitone = self.bins_per_semitone

    spectrogram_windows = []
    print("stacking the spectrogram")
    for mult in [1/(x+2) for x in range(undertones)]+list(range(1, overtones+1)):
        f_ref = 440  # arbitrary reference frequency
        hz = f_ref*mult
        interval = librosa.core.hz_to_midi(hz) - librosa.core.hz_to_midi(f_ref)

        int_bins = int(round((interval + offset)*bins_per_semitone))

        start = max(int_bins, 0)
        end = bin_count+int_bins
        spec_layer = input[:, :, start:end, :]

        print(mult, "start", start, "end", end, "shape", spec_layer.shape, end=" ")

        if int_bins < 0:
            spec_layer = tf.pad(spec_layer, ((0, 0), (0, 0), (-int_bins, 0), (0, 0)))

        spec_layer = tf.pad(spec_layer, ((0, 0), (0, 0), (0, bin_count-spec_layer.shape[2]), (0, 0)))

        print("padded shape", spec_layer.shape)

        spectrogram_windows.append(spec_layer)
    return tf.concat(spectrogram_windows, axis=-1)

def hconv2d(inputs, filters, kernel_size, undertones, overtones, bins_per_octave, **kwargs):
    # check for at least one resulting convolution
    assert overtones >= 1 or undertones > 1
    # assert kwargs["stride"] == (1, 1)
    # assert kwargs["padding"] == "same"

    mult_undertones = 1 / (np.arange(undertones) + 2)
    mult_overtones = np.arange(overtones) + 1
    harmonics = np.concatenate([mult_undertones, mult_overtones])
    shifts = np.round(bins_per_octave*np.log2(harmonics))

    with tf.name_scope('hconv2d'):
        convs = []
        for shift in shifts:
            sh = int(shift)
            if shift == 0:
                layer = inputs
            elif shift > 0:
                layer = inputs[:, :, sh:, :]
                # layer = tf.pad(layer, ((0, 0), (0, 0), (0, sh), (0, 0)))
            else:
                layer = inputs[:, :, :sh, :]
                # layer = tf.pad(layer, ((0, 0), (0, 0), (-sh, 0), (0, 0)))

            print(shift, layer.shape)

            layer = tf.layers.conv2d(layer, filters, kernel_size, **kwargs)

            if shift > 0:
                layer = tf.pad(layer, ((0, 0), (0, 0), (0, sh), (0, 0)))
            else:
                layer = tf.pad(layer, ((0, 0), (0, 0), (-sh, 0), (0, 0)))

            convs.append(layer)
        added = tf.math.add_n(convs)
    return added

def regularization(layer, args, training=None):
    if args.batchnorm:
        print("add batchnorm")
        layer = tf.layers.batch_normalization(layer, training=training)
    if args.dropout:
        print("add dropout")
        layer = tf.layers.dropout(layer, args.dropout, training=training)
    return layer

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


def add_layers_from_string(self, in_layer, layers_string):
    print("constructing", layers_string)
    if layers_string == "":
        return in_layer
    for layer in layers_string.split("--"):
        params = layer.split("_")
        layer_type = params.pop(0)
        p = {x[0]: x[1:] for x in params}

        if layer_type[:4] == "conv":
            activation = None
            if "a" in p:
                if p["a"] == "relu":
                    activation = tf.nn.relu
                if p["a"] == "tanh":
                    activation = tf.nn.tanh
            layer_fn = tf.layers.conv2d if layer_type[:6] == "conv2d" else tf.layers.conv1d
            in_layer = layer_fn(in_layer, filters=int(p["f"]), kernel_size=int(p["k"]),
                                strides=int(p["s"]), padding=p["P"], activation=activation)

        elif layer_type == "avgpool":
            in_layer = tf.layers.average_pooling1d(in_layer, pool_size=int(p["p"]), strides=int(p["s"]), padding=p["P"])

        elif layer_type == "maxpool":
            in_layer = tf.layers.max_pooling1d(in_layer, pool_size=int(p["p"]), strides=int(p["s"]), padding=p["P"])

        elif layer_type == "dropout":
            in_layer = tf.layers.dropout(in_layer, float(p["r"]), training=self.is_training)
        else:
            raise ValueError("Invalid layer in layers string")

        print(in_layer)

    return in_layer
# Dataset functions


def prepare_datasets(which, args, preload_fn, dataset_transform, dataset_transform_train, small_hooks_mf0=None, small_hooks=None, valid_hooks=None, test_hooks=None):
    timer = time.time()

    if small_hooks_mf0 is None:
        small_hooks_mf0 = [MetricsHook_mf0(), VisualOutputHook_mf0(True, True, False)]
    if small_hooks is None:
        small_hooks = [MetricsHook(), VisualOutputHook(True, True, False, False)]
    if valid_hooks is None:
        valid_hooks = [MetricsHook(), VisualOutputHook(False, False, True, True), SaveBestModelHook(args.logdir), CSVOutputWriterHook()]
    if test_hooks is None:
        test_hooks = [CSVOutputWriterHook()]
        if args.save_salience:
            test_hooks.append(SaveSaliencesHook())
    

    validation_datasets = []
    test_datasets = []
    train_data = []

    if args.predict:
        output_path = os.path.splitext(os.path.basename(args.predict))[0]
        uid = os.path.splitext(os.path.basename(args.predict))[0]
        # prepare audio
        audio = datasets.Audio(args.predict, uid)
        aa = datasets.AnnotatedAudio((None, uid), audio)
        preload_fn(aa)
        predict_dataset = datasets.AADataset([aa], args, dataset_transform)

        output_file = None
        if args.output_file:
            output_file = args.output_file
        test_datasets += [
            VD("predict", predict_dataset, 0, [CSVOutputWriterHook(output_path="./predict_outputs", output_file=output_file, output_format=args.output_format)]),
        ]
        return predict_dataset, test_datasets, []

    if datasets.musicnet_mir.prefix in which:
        musicnet_train, musicnet_test, musicnet_validation, musicnet_small_validation = datasets.musicnet_mir.prepare(preload_fn, threads=args.threads)
        musicnet_test_dataset = datasets.AADataset(musicnet_test, args, dataset_transform)
        musicnet_validation_dataset = datasets.AADataset(musicnet_validation, args, dataset_transform)
        musicnet_small_validation_dataset = datasets.AADataset(musicnet_small_validation, args, dataset_transform)

        validation_datasets += [
            VD(datasets.musicnet_mir.prefix, musicnet_validation_dataset, args.evaluate_every, valid_hooks),
            VD("small_"+datasets.musicnet_mir.prefix, musicnet_small_validation_dataset, args.evaluate_small_every, small_hooks_mf0),
        ]

        test_datasets += [
            VD(datasets.musicnet_mir.prefix, musicnet_test_dataset, 0, test_hooks),
            # VD(datasets.musicnet_mir.prefix, musicnet_validation_dataset, 0, test_hooks),
        ]

        train_data += musicnet_train

    if datasets.maps.prefix in which:
        maps_train, maps_test, maps_validation, maps_small_validation = datasets.maps.prepare(preload_fn, threads=args.threads)
        maps_test_dataset = datasets.AADataset(maps_test, args, dataset_transform)
        maps_validation_dataset = datasets.AADataset(maps_validation, args, dataset_transform)
        maps_small_validation_dataset = datasets.AADataset(maps_small_validation, args, dataset_transform)

        validation_datasets += [
            VD(datasets.maps.prefix, maps_validation_dataset, args.evaluate_every, valid_hooks),
            VD("small_"+datasets.maps.prefix, maps_small_validation_dataset, args.evaluate_small_every, small_hooks_mf0),
        ]

        test_datasets += [
            VD(datasets.maps.prefix, maps_test_dataset, 0, test_hooks),
            # VD(datasets.maps.prefix, maps_validation_dataset, 0, test_hooks),
        ]

        train_data += maps_train

    if datasets.medleydb.prefix in which or datasets.medleydb.prefix+"_mel4" in which:
        if datasets.medleydb.prefix+"_mel4" in which:
            annotation_type = "MELODY4"
        else:
            annotation_type = "MELODY2"
        medleydb_train, medleydb_test, medleydb_validation, medleydb_small_validation = datasets.medleydb.prepare(preload_fn, threads=args.threads, annotation_type=annotation_type)
        medleydb_test_dataset = datasets.AADataset(medleydb_test, args, dataset_transform)
        medleydb_validation_dataset = datasets.AADataset(medleydb_validation, args, dataset_transform)
        medleydb_small_validation_dataset = datasets.AADataset(medleydb_small_validation, args, dataset_transform)

        validation_datasets += [
            VD(datasets.medleydb.prefix, medleydb_validation_dataset, args.evaluate_every, valid_hooks),
        ]
        if datasets.medleydb.prefix+"_mel4" in which:
            validation_datasets += [
                VD("small_"+datasets.medleydb.prefix, medleydb_small_validation_dataset, args.evaluate_small_every, small_hooks_mf0),
            ]
        else:
            validation_datasets += [
                VD("small_"+datasets.medleydb.prefix, medleydb_small_validation_dataset, args.evaluate_small_every, small_hooks),
            ]

        test_datasets += [
            VD(datasets.medleydb.prefix, medleydb_test_dataset, 0, test_hooks),
            VD(datasets.medleydb.prefix, medleydb_validation_dataset, 0, test_hooks),
        ]

        train_data += medleydb_train

    if datasets.mdb_melody_synth.prefix in which:
        mdb_melody_synth_train, mdb_melody_synth_test, mdb_melody_synth_validation, _ = datasets.mdb_melody_synth.prepare(preload_fn, subsets=("test", "validation"))
        mdb_melody_synth_test_dataset = datasets.AADataset(mdb_melody_synth_test, args, dataset_transform)
        mdb_melody_synth_validation_dataset = datasets.AADataset(mdb_melody_synth_validation, args, dataset_transform)
        validation_datasets += [
            VD(datasets.mdb_melody_synth.prefix, mdb_melody_synth_validation_dataset, args.evaluate_every, valid_hooks),
        ]
        test_datasets += [
            VD(datasets.mdb_melody_synth.prefix, mdb_melody_synth_test_dataset, 0, test_hooks),
            VD(datasets.mdb_melody_synth.prefix, mdb_melody_synth_validation_dataset, 0, test_hooks),
        ]
        train_data += mdb_melody_synth_train

    if datasets.mdb_stem_synth.prefix in which:
        mdb_stem_synth_train, mdb_stem_synth_test, mdb_stem_synth_validation, mdb_stem_synth_small_validation = datasets.mdb_stem_synth.prepare(preload_fn)
        mdb_stem_synth_small_validation_dataset = datasets.AADataset(mdb_stem_synth_small_validation, args, dataset_transform)
        mdb_stem_synth_test_dataset = datasets.AADataset(mdb_stem_synth_test, args, dataset_transform)
        mdb_stem_synth_validation_dataset = datasets.AADataset(mdb_stem_synth_validation, args, dataset_transform)
        validation_datasets += [
            VD("small_"+datasets.mdb_stem_synth.prefix, mdb_stem_synth_small_validation_dataset, args.evaluate_small_every, small_hooks),
            VD(datasets.mdb_stem_synth.prefix, mdb_stem_synth_validation_dataset, args.evaluate_every, valid_hooks),
        ]
        test_datasets += [
            VD(datasets.mdb_stem_synth.prefix, mdb_stem_synth_test_dataset, 0, test_hooks),
            VD(datasets.mdb_stem_synth.prefix, mdb_stem_synth_validation_dataset, 0, test_hooks),
        ]
        train_data += mdb_stem_synth_train

    if datasets.mdb_mf0_synth.prefix in which:
        _, _, mdb_mf0_synth_small_validation = datasets.mdb_mf0_synth.prepare(preload_fn)
        mdb_mf0_synth_small_validation_dataset = datasets.AADataset(mdb_mf0_synth_small_validation, args, dataset_transform)
        validation_datasets += [
            VD("small_"+datasets.mdb_mf0_synth.prefix, mdb_mf0_synth_small_validation_dataset, args.evaluate_small_every, small_hooks_mf0),
        ]
    
    if datasets.wjazzd.prefix in which:
        wjazzd_train, wjazzd_test, wjazzd_validation, wjazzd_small_validation = datasets.wjazzd.prepare(preload_fn, subsets=("test", "validation"))
        wjazzd_test_dataset = datasets.AADataset(wjazzd_test, args, dataset_transform)
        wjazzd_validation_dataset = datasets.AADataset(wjazzd_validation, args, dataset_transform)
        wjazzd_small_validation_dataset = datasets.AADataset(wjazzd_small_validation, args, dataset_transform)
        validation_datasets += [
            VD("small_"+datasets.wjazzd.prefix, wjazzd_small_validation_dataset, args.evaluate_small_every, small_hooks),
            VD(datasets.wjazzd.prefix, wjazzd_validation_dataset, args.evaluate_small_every, valid_hooks),
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
        validation_datasets.append(VD("small_"+datasets.orchset.prefix, orchset_small_validation_dataset, args.evaluate_small_every, small_hooks))
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
        print("Warning: using automatically selected train_dataset")
        train_dataset = (test_datasets+validation_datasets)[0].dataset

    print("datasets ready in {:.2f}s".format(time.time() - timer))

    return train_dataset, test_datasets, validation_datasets

# Main functions


def main(argv, construct, parse_args):
    args = parse_args(argv)
    # Construct the network
    network, train_dataset, validation_datasets, test_datasets = construct(args)

    if not (args.evaluate and network.restored and not args.rewind) and not args.predict:
        try:
            network.train(train_dataset, validation_datasets, save_every_n_batches=10000)
            network.save(args.checkpoint)
        except KeyboardInterrupt:
            network.save(args.checkpoint)
            sys.exit()

    if args.evaluate:
        for vd in test_datasets:
            print("{} evaluation".format(vd.name))
            network.evaluate(vd)

        print(evaluation.summary("test", os.path.join(args.logdir, args.checkpoint+"-f0-outputs")))

    if args.predict:
        for vd in test_datasets:
            print("{}".format(vd.name))
            network.evaluate(vd)
