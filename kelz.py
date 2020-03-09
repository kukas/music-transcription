import tensorflow as tf
import numpy as np

import datasets
from model import NetworkMultif0, AdjustVoicingHook, MetricsHook_mf0, SaveBestModelHook, CSVOutputWriterHook, VisualOutputHook_mf0
from collections import namedtuple
import sys
import common

import librosa

from tensorflow.python.ops import array_ops

from madmom.audio.spectrogram import LogarithmicFilterbank, LogarithmicFilteredSpectrogram
from madmom.audio.signal import FramedSignal


def create_model(self, args):
    args_context_size = int(self.context_width/self.spectrogram_hop_size)

    if args.activation is not None:
        activation = getattr(tf.nn, args.activation)

    with tf.name_scope('kelz'):
        layer = self.spectrogram
        print("self.spectrogram shape", self.spectrogram.shape)

        layer = tf.layers.conv2d(layer, args.filters, (3, 3), (1, 1), "valid", activation=None, use_bias=False)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = activation(layer)

        layer = tf.layers.conv2d(layer, args.filters, (3, 3), (1, 1), "valid", activation=None, use_bias=False)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = activation(layer)

        layer = tf.layers.max_pooling2d(layer, (1,2), (1,2))
        layer = tf.layers.dropout(layer, args.dropout, training=self.is_training)

        layer = tf.layers.conv2d(layer, args.filters, (1, 3), (1, 1), "valid", activation=None, use_bias=False)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = activation(layer)

        layer = tf.layers.conv2d(layer, args.filters, (1, 3), (1, 1), "valid", activation=None, use_bias=False)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = activation(layer)

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

        print("note_logits shape", self.note_logits.shape)

        self.voicing_threshold = tf.Variable(0.5, trainable=False)
        tf.summary.scalar("model/voicing_threshold", self.voicing_threshold)

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
    parser.add_argument("--stacks", type=int, help="Stacks")
    parser.add_argument("--conv_range", type=int, help="Stack kernel size in frequency axis")
    parser.add_argument("--undertone_stacking", type=int, help="Undertone stacking in the model")
    parser.add_argument("--overtone_stacking", type=int, help="Overtone stacking in the model")
    parser.add_argument("--activation", type=str, help="Activation function for the convolution stack")
    # context
    parser.add_argument("--conv_ctx", nargs="+", type=int, help="Stack kernel sizes in time axis")
    parser.add_argument("--dilations", nargs="+", type=int, help="Dilation rate for the convolutions")
    parser.add_argument("--last_conv_kernel", nargs=2, type=int)
    # regularization
    parser.add_argument("--batchnorm", type=int)
    parser.add_argument("--dropout", type=float)

    args = parser.parse_args(argv)

    hop_length = 441 * 4  # 25 fps
    # context_width: 10*hop_length
    defaults = {
        # Change some of the common defaults
        "samplerate": 44100, "context_width": 2*hop_length, "annotations_per_window": 1, "hop_size": 1, "frame_width": hop_length,
        "note_range": 88, "min_note": 21, "evaluate_every": 5000, "evaluate_small_every": 1000, "annotation_smoothing": 0.5, "batch_size": 128,
        "batch_size_evaluation": 1024,
        "bins_per_semitone": 1,
        "datasets": ["maps"],
        # Model specific defaults
        "learning_rate_decay_steps": 10000,
        "learning_rate_decay": 0.8,
        "spectrogram": "kelz",
        "architecture": "kelz",
        "filters": 16,
        "stacks": 10,
        "conv_range": 3,
        "undertone_stacking": 0,
        "overtone_stacking": 1,
        "activation": "relu",
        "conv_ctx": [1],
        "dilations": [1],
        "last_conv_kernel": [1, 1],
        "dropout": 0.25,
    }
    specified_args = common.argument_defaults(args, defaults)
    common.name(args, specified_args, "mf0")

    return args

def construct(args):
    network = NetworkMultif0(args)

    # https://github.com/rainerkelz/framewise_2016/blob/master/datasets.py
    if args.spectrogram == "kelz":
        FMIN = 30
        FMAX = 8000
        NUMBANDS = 48
        HOP_LENGTH = args.frame_width
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
        train_dataset, test_datasets, validation_datasets = common.prepare_datasets(args.datasets, args, preload_fn, dataset_transform, dataset_transform_train, small_hooks_mf0=small_hooks, valid_hooks=valid_hooks)

        # if not args.voicing:
        #     for vd in validation_datasets:
        #         if not vd.name.startswith("small_"):
        #             vd.hooks.append(AdjustVoicingHook())

        network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes, spectrogram_info=spectrogram_info)

    return network, train_dataset, validation_datasets, test_datasets


if __name__ == "__main__":
    common.main(sys.argv[1:], construct, parse_args)
