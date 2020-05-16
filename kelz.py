import tensorflow as tf
import numpy as np

import datasets
from model import NetworkMultif0, AdjustVoicingHook, MetricsHook_mf0, SaveBestModelHook, CSVBatchOutputWriterHook_mf0, VisualOutputHook_mf0
from collections import namedtuple
import sys
import common

import librosa

from tensorflow.python.ops import array_ops

from madmom.audio.spectrogram import LogarithmicFilterbank, LogarithmicFilteredSpectrogram
from madmom.audio.signal import FramedSignal

def create_model(self, args):
    
    if args.spectrogram_undertone_stacking > 0 or args.spectrogram_overtone_stacking > 1:
        spectrogram = common.harmonic_stacking(self, self.spectrogram, args.spectrogram_undertone_stacking, args.spectrogram_overtone_stacking, bin_count=229, bins_per_semitone=4)
    else:
        spectrogram = self.spectrogram[:, :, :229, :]

    if args.activation is not None:
        activation = getattr(tf.nn, args.activation)

    print("self.spectrogram shape", self.spectrogram.shape)
    print("spectrogram shape", spectrogram.shape)

    with tf.name_scope('model'):
        if args.architecture == "allconv":
            layer = spectrogram

            layer = tf.layers.conv2d(layer, args.filters, (3, 3), (1, 1), "valid", activation=None, use_bias=False)
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = activation(layer)

            if args.undertone_stacking > 0 or args.overtone_stacking > 1:
                layer = common.harmonic_stacking(self, layer, args.undertone_stacking, args.overtone_stacking, bin_count=227, bins_per_semitone=4)

            layer = tf.layers.conv2d(layer, args.filters, (3, 3), (1, 1), "valid", activation=None, use_bias=False)
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = activation(layer)

            if args.undertone_stacking > 0 or args.overtone_stacking > 1:
                layer = common.harmonic_stacking(self, layer, args.undertone_stacking, args.overtone_stacking, bin_count=225, bins_per_semitone=4)

            layer = tf.layers.max_pooling2d(layer, (1,2), (1,2))
            layer = tf.layers.dropout(layer, args.dropout, training=self.is_training)
            
            layer = tf.layers.conv2d(layer, args.filters, (1, 3), (1, 1), "valid", activation=None, use_bias=False)
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = activation(layer)

            if args.undertone_stacking > 0 or args.overtone_stacking > 1:
                layer = common.harmonic_stacking(self, layer, args.undertone_stacking, args.overtone_stacking, bin_count=110, bins_per_semitone=2)

            layer = tf.layers.conv2d(layer, args.filters, (1, 3), (1, 1), "valid", activation=None, use_bias=False)
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = activation(layer)

            if args.undertone_stacking > 0 or args.overtone_stacking > 1:
                layer = common.harmonic_stacking(self, layer, args.undertone_stacking, args.overtone_stacking, bin_count=108, bins_per_semitone=2)

            layer = tf.layers.max_pooling2d(layer, (1,2), (1,2))
            layer = tf.layers.dropout(layer, args.dropout, training=self.is_training)

            layer = tf.layers.conv2d(layer, args.filters*2, (1, 25), (1, 1), "valid", activation=None, use_bias=False)
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = activation(layer)

            layer = tf.layers.conv2d(layer, args.filters*4, (1, 25), (1, 1), "valid", activation=None, use_bias=False)
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = activation(layer)

            layer = tf.layers.conv2d(layer, args.note_range, (1, 1), (1, 1), "valid", activation=None, use_bias=False)
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = tf.layers.average_pooling2d(layer, (1,6), (1,6))

            # layer.shape (?, 1, 1, 88) = (batch_size, annot_per_window, freq, channels)
            layer = tf.squeeze(layer, 2)
            self.note_logits = layer

        if args.architecture == "vggnet":
            layer = spectrogram

            print("spectrogram", layer.shape)

            layer = tf.layers.conv2d(layer, args.filters, (3, 3), (1, 1), "same", activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = activation(layer)
            
            print("conv", layer.shape)

            if args.undertone_stacking > 0 or args.overtone_stacking > 1:
                layer = common.harmonic_stacking(self, layer, args.undertone_stacking, args.overtone_stacking, bin_count=229, bins_per_semitone=4)

            layer = tf.layers.conv2d(layer, args.filters, (3, 3), (1, 1), "valid", activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = activation(layer)
            
            print("conv2", layer.shape)

            if args.undertone_stacking > 0 or args.overtone_stacking > 1:
                layer = common.harmonic_stacking(self, layer, args.undertone_stacking, args.overtone_stacking, bin_count=227, bins_per_semitone=4)

            layer = tf.layers.max_pooling2d(layer, (1, 2), (1, 2))
            layer = tf.layers.dropout(layer, args.dropout, training=self.is_training)

            print("maxpool", layer.shape)

            layer = tf.layers.conv2d(layer, args.filters*2, (3, 3), (1, 1), "valid", activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = activation(layer)

            layer = tf.layers.max_pooling2d(layer, (1, 2), (1, 2))
            layer = tf.layers.dropout(layer, args.dropout, training=self.is_training)

            print("conv3", layer.shape)
            layer = tf.layers.flatten(layer)
            print("flatten", layer.shape)
            # in the implementation in framewise_2016 repository I left 512 hidden units fixed
            layer = tf.layers.dense(layer, args.filters*16, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = activation(layer)
            print("fc1", layer.shape)
            layer = tf.layers.dropout(layer, 0.5, training=self.is_training)
            layer = tf.layers.dense(layer, args.note_range*args.annotations_per_window, kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))
            print("fc2", layer.shape)
            layer = tf.reshape(layer, [-1, args.annotations_per_window, args.note_range])
            self.note_logits = layer

    print("note_logits shape", self.note_logits.shape)

    self.voicing_threshold = tf.Variable(0.5, trainable=False)

    self.loss = common.loss_mf0(self, args)
    self.est_notes = tf.constant(0) # placeholder, we compute est_notes on cpu
    self.training = common.optimizer(self, args)

def parse_args(argv):
    parser = common.common_arguments_parser()
    # Model specific arguments
    # input
    parser.add_argument("--spectrogram", type=str, help="Spectrogram method")
    # model
    parser.add_argument("--architecture", type=str, help="Model architecture")
    parser.add_argument("--filters", type=int, help="Filters in convolutions")
    parser.add_argument("--spectrogram_undertone_stacking", type=int, help="Undertone stacking in the spectrogram")
    parser.add_argument("--spectrogram_overtone_stacking", type=int, help="Overtone stacking in the spectrogram")
    parser.add_argument("--undertone_stacking", type=int, help="Undertone stacking in the model")
    parser.add_argument("--overtone_stacking", type=int, help="Overtone stacking in the model")
    parser.add_argument("--activation", type=str, help="Activation function for the convolution stack")
    # regularization
    parser.add_argument("--dropout", type=float)

    args = parser.parse_args(argv)

    hop_length = 441 * 4  # 25 fps
    # context_width: 10*hop_length
    defaults = {
        # Change some of the common defaults
        "samplerate": 44100, "context_width": 2*hop_length, "annotations_per_window": 1, "hop_size": 1, "frame_width": hop_length,
        "note_range": 88, "min_note": 21, "evaluate_every": 5000, "evaluate_small_every": 1000, "annotation_smoothing": 0.0, "batch_size": 128,
        "batch_size_evaluation": 1024,
        "bins_per_semitone": 1,
        "datasets": ["maps"],
        # Model specific defaults
        "learning_rate_decay_steps": 10000,
        "learning_rate_decay": 0.8,
        "spectrogram": "kelz",
        "architecture": "allconv",
        "filters": 16,
        "undertone_stacking": 0,
        "overtone_stacking": 1,
        "spectrogram_undertone_stacking": 0,
        "spectrogram_overtone_stacking": 1,
        "activation": "relu",
        "dropout": 0.25,
    }
    specified_args = common.argument_defaults(args, defaults)
    common.name(args, specified_args, "kelz")

    return args

def construct(args):
    network = NetworkMultif0(args)

    HOP_LENGTH = args.frame_width
    # https://github.com/rainerkelz/framewise_2016/blob/master/datasets.py
    if args.spectrogram == "kelz":
        FMIN = 30
        FMAX = 8000
        NUMBANDS = 48
        def spectrogram_function(audio, samplerate):
            audio_options = dict(
                num_channels=1,
                sample_rate=samplerate,
                filterbank=LogarithmicFilterbank,
                frame_size=4096,
                fft_size=4096,
                hop_size=HOP_LENGTH,
                num_bands=NUMBANDS,
                fmin=FMIN,
                fmax=FMAX,
                fref=440.0,
                norm_filters=True,
                unique_filters=True,
                circular_shift=False,
                norm=True
            )
            x = LogarithmicFilteredSpectrogram(audio, **audio_options)
            x = x.T
            x = x / np.max(x)
            x = np.expand_dims(x, 0)
            return (np.array(x)*65535).astype(np.uint16)
        
        N_BINS = 229

        spectrogram_thumb = "kelz-fmin{}-fmax{}-bands{}-hop{}-uint16".format(FMIN, FMAX, NUMBANDS, HOP_LENGTH)
        spectrogram_info = (1, N_BINS, HOP_LENGTH, FMIN)

    elif args.spectrogram == "cqt_kelz":
        FMIN = 32.7
        BINS_PER_OCTAVE = 48
        N_BINS = BINS_PER_OCTAVE*5
        top_db = 110
        filter_scale = 1.0

        def spectrogram_function(audio, samplerate):
            cqt = librosa.cqt(audio, sr=samplerate, hop_length=HOP_LENGTH, fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, filter_scale=filter_scale)

            log_cqt = (librosa.core.amplitude_to_db(np.abs(cqt), ref=np.max, top_db=top_db) / top_db) + 1.0
            log_cqt = np.expand_dims(log_cqt, 0)
            return (log_cqt*65535).astype(np.uint16)

        spectrogram_thumb = "cqt-fmin{}-oct{}-octbins{}-hop{}-db{}-fs{}-uint16".format(FMIN, N_BINS/BINS_PER_OCTAVE, BINS_PER_OCTAVE, HOP_LENGTH, top_db, filter_scale)
        spectrogram_info = (1, N_BINS, HOP_LENGTH, FMIN)

    else:
        spectrogram_function, spectrogram_thumb, spectrogram_info = common.spectrograms(args)

    # save spectrogram_thumb to hyperparams
    args.spectrogram_thumb = spectrogram_thumb

    with network.session.graph.as_default():
        def preload_fn(aa):
            annot_path, uid = aa.annotation
            if uid.startswith("mdb_"):
                uid = uid + "_mel4"
            if uid.startswith("maps_"):
                aa.annotation = datasets.Annotation.from_midi(annot_path, uid, hop_samples=args.frame_width*args.samplerate/44100, unique_mf0=True)
            else:
                aa.annotation = datasets.Annotation.from_time_series(annot_path, uid, hop_samples=args.frame_width*args.samplerate/44100, unique_mf0=True)
            aa.audio.load_resampled_audio(args.samplerate).load_spectrogram(spectrogram_function, spectrogram_thumb, spectrogram_info[2])

        def dataset_transform(tf_dataset, dataset):
            return tf_dataset.map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size_evaluation).prefetch(10)

        def dataset_transform_train(tf_dataset, dataset):
            return tf_dataset.shuffle(10**5).map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size).prefetch(10)
        small_hooks = [MetricsHook_mf0(), VisualOutputHook_mf0()]
        valid_hooks = [MetricsHook_mf0(), SaveBestModelHook(args.logdir, "Accuracy")]
        test_hooks = [MetricsHook_mf0(write_summaries=True, print_detailed=False, split="test"), CSVBatchOutputWriterHook_mf0(output_reference=True)]
        train_dataset, test_datasets, validation_datasets = common.prepare_datasets(
            args.datasets, args, preload_fn, dataset_transform, dataset_transform_train, 
            small_hooks_mf0=small_hooks, valid_hooks=valid_hooks, test_hooks=test_hooks)

        network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes, spectrogram_info=spectrogram_info)

    return network, train_dataset, validation_datasets, test_datasets


if __name__ == "__main__":
    common.main(sys.argv[1:], construct, parse_args)
