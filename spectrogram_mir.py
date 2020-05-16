import tensorflow as tf
import numpy as np

import datasets
from model import *
from collections import namedtuple
import sys
import common

import librosa

from tensorflow.python.ops import array_ops


def create_model(self, args):
    if args.spectrogram == "cqt":
        spec_bin_count = 360
        spec_bins_per_semitone = 5
    if args.spectrogram == "YunNingHung_cqt":
        spec_bin_count = 88
        spec_bins_per_semitone = 1

    if args.spectrogram_undertone_stacking > 0 or args.spectrogram_overtone_stacking > 1:
        spectrogram = common.harmonic_stacking(self, self.spectrogram, args.spectrogram_undertone_stacking, args.spectrogram_overtone_stacking,
                                               bin_count=spec_bin_count, bins_per_semitone=spec_bins_per_semitone)
    else:
        spectrogram = self.spectrogram
        if args.spectrogram == "cqt":
            spectrogram = self.spectrogram[:, :, :spec_bin_count, :]

    args_context_size = int(self.context_width/self.spectrogram_hop_size)

    if args.activation is not None:
        activation = getattr(tf.nn, args.activation)

    with tf.name_scope('model_pitch'):
        layer = spectrogram
        print("self.spectrogram shape", self.spectrogram.shape)
        print("spectrogram shape", spectrogram.shape)

        if args.architecture.startswith("baseline"):
            # layer = tf.layers.conv2d(layer, args.filters, (args.conv_ctx[0], args.conv_range), strides=(1, 5), padding="same", activation=None)
            #layer = activation(layer)
            #layer = tf.layers.average_pooling2d(layer, (5, 1), (5, 1))
            layer = tf.layers.flatten(layer)
            layer = tf.layers.dense(layer, 100, use_bias=(not args.batchnorm))
            if args.batchnorm:
                layer = tf.layers.batch_normalization(layer, training=self.is_training)

            layer = activation(layer)
            layer = tf.layers.dense(layer, args.note_range*args.annotations_per_window)
            layer = tf.reshape(layer, (-1, args.annotations_per_window, args.note_range))
            self.note_logits = layer
            # layer_cut = layer[:, args_context_size:-args_context_size, :, :]
            # self.note_logits = tf.squeeze(layer_cut, -1)
            print("note_logits shape", self.note_logits.shape)

        if args.architecture.startswith("LY"):
            # batch_size, annotations_per_wind, time, freq
            def conv_block(self, layer, args, channels, kernel, time_padding):
                layer = tf.pad(layer, ((0, 0), (time_padding, time_padding), (0, 0), (0, 0)))
                layer = tf.layers.conv2d(layer, channels, kernel, padding="valid", activation=None, use_bias=False)
                if args.batchnorm:
                    layer = tf.layers.batch_normalization(layer, training=self.is_training)
                layer = activation(layer)

                return layer
            print(layer.shape)
            layer = conv_block(self, layer, args, args.filters, (7, spectrogram.shape[2]), 3)
            print(layer.shape)
            layer = tf.layers.max_pooling2d(layer, (3, 1), (3, 1))
            print(layer.shape)
            layer = conv_block(self, layer, args, args.filters, (7, 1), 3)
            print(layer.shape)
            layer = tf.layers.max_pooling2d(layer, (3, 1), (3, 1))
            print(layer.shape)
            layer = conv_block(self, layer, args, 16*args.filters, (1, 1), 0)
            print(layer.shape)
            layer = conv_block(self, layer, args, 16*args.filters, (1, 1), 0)
            print(layer.shape)
            layer = tf.layers.conv2d(layer, self.note_range, (1, 1), padding="valid", activation=None)
            print(layer.shape)
            # squeeze frequency dimension
            layer = tf.squeeze(layer, 2)
            
            self.note_logits = layer
            print("note_logits shape", self.note_logits.shape)

        if args.architecture.startswith("deep_hcnn"):
            assert len(args.conv_ctx) <= args.stacks
            # Prepare kernel sizes (time axis = audio context)
            args_ctx = np.abs(args.conv_ctx)
            args_dils = np.abs(args.dilations)
            ctxs = np.array([args_ctx[i] if i < len(args_ctx) else args_ctx[-1] for i in range(args.stacks)])
            dils = np.array([args_dils[i] if i < len(args_dils) else args_dils[-1] for i in range(args.stacks)])
            if args.conv_ctx[0] < 0:
                ctxs = np.array(list(reversed(ctxs)))
            if args.dilations[0] < 0:
                dils = np.array(list(reversed(dils)))
            print(ctxs)

            # Cut the unnecessary context
            needed_context_size = int(np.sum(np.ceil((ctxs-1)/2)) + np.ceil((args.last_conv_kernel[0]-1)/2))
            actual_context_size = args_context_size
            print("input context", args_context_size, "actual needed context", needed_context_size)
            if args_context_size < needed_context_size:
                print("Warning: provided context is shorter than the needed context field of the network")
            elif args_context_size > needed_context_size:
                if args.cut_context:
                    print("Cutting the unnecessary context {} --> ".format(layer.shape), end="")
                    diff = args_context_size - needed_context_size
                    layer = layer[:, diff:-diff, :, :]
                    actual_context_size -= diff
                    print(layer.shape, "context now:", actual_context_size)

            skip = None
            for i, conv_ctx, dil in zip(range(args.stacks), ctxs, dils):
                kernel = (conv_ctx, args.conv_range)

                if i > 0 and args.faster_hcnn:
                    print("add hconv2d {} filters, {} kernel".format(args.filters, kernel))
                    layer = common.hconv2d(
                        layer, args.filters, kernel,
                        args.undertone_stacking, args.overtone_stacking, 60, # bins per octave
                        padding="same", activation=None, dilation_rate=(dil, 1), use_bias=bool(args.use_bias))
                    print(layer.shape)
                else:
                    print("add conv2d {} filters, {} kernel".format(args.filters, kernel))
                    layer = tf.layers.conv2d(
                        layer, args.filters, kernel, (1, 1), 
                        padding="same", activation=None, dilation_rate=(dil, 1), use_bias=bool(args.use_bias))
                    print(layer.shape)

                layer = common.regularization(layer, args, training=self.is_training)
                layer = activation(layer)

                if (not args.faster_hcnn) and (args.undertone_stacking > 0 or args.overtone_stacking > 1) and i < args.stacking_until:
                    print("harmonic stacking {} --> ".format(layer.shape), end="")
                    layer = common.harmonic_stacking(self, layer, args.undertone_stacking, args.overtone_stacking, bin_count=360, bins_per_semitone=5)
                    print(layer.shape)


                if i < args.stacks - args.residual_end and i % args.residual_hop == 0:
                    if skip is None:
                        print(".- begin residual connection")
                    else:
                        if args.residual_op == "add":
                            print("|- adding residual connection")
                            layer += skip
                        if args.residual_op == "concat":
                            print("|- concatenating residual connection")
                            layer = tf.concat([skip, layer], -1)
                    skip = layer

            if args.last_pooling == "globalavg":
                layer = tf.layers.average_pooling2d(layer, (1, 360), (1, 360))
            if args.last_pooling == "avg":
                layer = tf.layers.average_pooling2d(layer, (1, 5), (1, 5))
            if args.last_pooling == "max":
                layer = tf.layers.max_pooling2d(layer, (1, 5), (1, 5))
            if args.last_pooling == "maxoct":
                layer = tf.layers.max_pooling2d(layer, (1, 60), (1, 60))

            if args.faster_hcnn:
                print("add last hconv2d {} filters, {} kernel".format(args.filters, kernel))
                layer = common.hconv2d(
                    layer, args.note_range, args.last_conv_kernel,
                    args.undertone_stacking, args.overtone_stacking, 12,  # bins per semitone
                    padding="valid", activation=None, use_bias=bool(args.use_bias))
                print(layer.shape)
            else:
                print("add last conv2d {} filters, {} kernel".format(args.filters, kernel))
                layer = tf.layers.conv2d(
                    layer, args.note_range, args.last_conv_kernel, (1, 1),
                    padding="valid", activation=None, use_bias=bool(args.use_bias))
                print(layer.shape)


            if actual_context_size > 0:
                layer = layer[:, actual_context_size:-actual_context_size, :, :]

            self.note_logits = tf.squeeze(layer, 2)
            print("note_logits shape", self.note_logits.shape)

    if args.class_weighting:
        weights = self.class_weights
    else:
        weights = None

    self.loss = common.loss_mir(self, args, weights=weights)
    self.est_notes = tf.constant(0) # placeholder, we compute est_notes on cpu
    self.training = common.optimizer(self, args)



def parse_args(argv):
    parser = common.common_arguments_parser()
    # Model specific arguments
    # input
    parser.add_argument("--spectrogram", type=str, help="Spectrogram method")
    parser.add_argument("--spectrogram_top_db", type=float, help="Spectrogram top_db")
    parser.add_argument("--spectrogram_filter_scale", type=float, help="Spectrogram filter_scale")
    parser.add_argument("--spectrogram_undertone_stacking", type=int, help="spectrogram undertone stacking")
    parser.add_argument("--spectrogram_overtone_stacking", type=int, help="spectrogram overtone stacking")
    parser.add_argument("--cut_context", type=int, help="Cut unnecessary context, doesn't work with dilations!")
    # model
    parser.add_argument("--architecture", type=str, help="Model architecture")
    parser.add_argument("--faster_hcnn", type=int, help="HCNN implementation")
    parser.add_argument("--use_bias", type=int, help="use bias in conv2d")
    parser.add_argument("--class_weighting", type=int, help="use class weighting")

    parser.add_argument("--filters", type=int, help="Filters in convolutions")
    parser.add_argument("--stacks", type=int, help="Stacks")
    parser.add_argument("--conv_range", type=int, help="Stack kernel size in frequency axis")
    parser.add_argument("--undertone_stacking", type=int, help="Undertone stacking in the model")
    parser.add_argument("--overtone_stacking", type=int, help="Overtone stacking in the model")
    parser.add_argument("--stacking_until", type=int, help="Harmonic stacking in the model until Nth layer")
    parser.add_argument("--activation", type=str, help="Activation function for the convolution stack")
    # context
    parser.add_argument("--conv_ctx", nargs="+", type=int, help="Stack kernel sizes in time axis")
    parser.add_argument("--dilations", nargs="+", type=int, help="Dilation rate for the convolutions")
    parser.add_argument("--last_conv_kernel", nargs=2, type=int)
    parser.add_argument("--last_pooling", type=str)
    # residual
    parser.add_argument("--residual_hop", type=int, help="Size of one block around which there is a residual connection")
    parser.add_argument("--residual_end", type=int, help="No residual connection in last N layers")
    parser.add_argument("--residual_op", type=str, help="Residual connection operation (add for ResNet, concat for DenseNet)")
    # regularization
    parser.add_argument("--batchnorm", type=int)
    parser.add_argument("--dropout", type=float)

    args = parser.parse_args(argv)

    # hop_length = 256
    hop_length = 512 # FRAME-LEVEL INSTRUMENT RECOGNITION BY TIMBRE AND PITCH
    defaults = {
        # Change some of the common defaults
        "samplerate": 44100, "context_width": 4*hop_length, "annotations_per_window": 1, "hop_size": 1, "frame_width": hop_length,
        "note_range": 11, "min_note": 0, "evaluate_every": 30000, "evaluate_small_every": 1000, "annotation_smoothing": 0.0, "batch_size": 32,
        "bins_per_semitone": 1,
        "unvoiced_loss_weight": 1.0,
        "datasets": ["musicnet_mir"],
        # Model specific defaults
        "learning_rate_decay_steps": 10000,
        "learning_rate_decay": 0.8,
        "spectrogram": "YunNingHung_cqt",
        "spectrogram_top_db": 110,
        "spectrogram_filter_scale": 1.0,
        "spectrogram_undertone_stacking": 1,
        "spectrogram_overtone_stacking": 5,
        "cut_context": 1,
        "architecture": "baseline",
        "faster_hcnn": 0,
        "use_bias": 1,
        "class_weighting": 1,
        "filters": 12,
        "stacks": 6,
        "conv_range": 3,
        "undertone_stacking": 1,
        "overtone_stacking": 3,
        "stacking_until": 999,
        "activation": "relu",
        "conv_ctx": [1],
        "dilations": [1],
        "last_conv_kernel": [1, 72],
        "last_pooling": "avg",
        "residual_hop": 1,
        "residual_end": 0,
        "residual_op": "add",
        "batchnorm": 0,
        "dropout": 0.3,
    }
    specified_args = common.argument_defaults(args, defaults)
    common.name(args, specified_args, "mir")

    return args

def construct(args):
    network = NetworkMultiInstrumentRecognition(args)

    with network.session.graph.as_default():
        if args.spectrogram == "YunNingHung_cqt":
            HOP_LENGTH = args.frame_width
            FMIN = 27.5
            BINS_PER_OCTAVE = 12
            N_BINS = 88
            top_db = args.spectrogram_top_db
            filter_scale = args.spectrogram_filter_scale

            def spectrogram_function(audio, samplerate):
                print(np.min(audio), np.max(audio))
                cqt = librosa.cqt(audio, sr=samplerate, hop_length=HOP_LENGTH, fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, filter_scale=filter_scale)
                print(np.min(cqt), np.max(cqt))

                log_cqt = (librosa.core.amplitude_to_db(np.abs(cqt), ref=np.max, top_db=top_db) / top_db) + 1.0
                log_cqt = np.expand_dims(log_cqt, 0)
                return (log_cqt*65535).astype(np.uint16)

            spectrogram_thumb = "YunNingHung_cqt-fmin{}-oct{}-octbins{}-hop{}-db{}-fs{}-uint16".format(FMIN, N_BINS/BINS_PER_OCTAVE, BINS_PER_OCTAVE, HOP_LENGTH, top_db, filter_scale)
            spectrogram_info = (1, N_BINS, HOP_LENGTH, FMIN)
        else:
            spectrogram_function, spectrogram_thumb, spectrogram_info = common.spectrograms(args)
            # save spectrogram_thumb to hyperparams

        args.spectrogram_thumb = spectrogram_thumb

        # all instruments in MusicNet
        # mapping between MIDI instrument and position in output probability tensor
        instrument_mappings = {
            1:  {"id": 0, "instrument": "piano"},
            7:  {"id": 1, "instrument": "harpsichord"},
            41: {"id": 2, "instrument": "violin"},
            42: {"id": 3, "instrument": "viola"},
            43: {"id": 4, "instrument": "cello"},
            44: {"id": 5, "instrument": "contrabass"},
            61: {"id": 6, "instrument": "french horn"},
            69: {"id": 7, "instrument": "oboe"},
            71: {"id": 8, "instrument": "bassoon"},
            72: {"id": 9, "instrument": "clarinet"},
            74: {"id": 10, "instrument": "flute"},
        }

        def preload_fn(aa):
            annot_path, uid = aa.annotation
            if uid.startswith("musicnet_mir"):
                aa.annotation = datasets.Annotation.from_musicnet_csv(annot_path, uid, hop_samples=args.frame_width*args.samplerate/44100, unique_mf0=True, instrument_mappings=instrument_mappings)
            aa.audio.load_resampled_audio(args.samplerate).load_spectrogram(spectrogram_function, spectrogram_thumb, spectrogram_info[2])

        def dataset_transform(tf_dataset, dataset):
            return tf_dataset.map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size_evaluation).prefetch(10)

        def dataset_transform_train(tf_dataset, dataset):
            return tf_dataset.shuffle(10**5).map(dataset.prepare_example, num_parallel_calls=args.threads).batch(args.batch_size).prefetch(10)
        small_hooks = [MetricsHook_mir(instrument_mappings), VisualOutputHook_mir()]
        valid_hooks = [AdjustVoicingHook_mir(), MetricsHook_mir(instrument_mappings), SaveBestModelHook(args.logdir, "micro f1"), BatchOutputWriterHook_mir(split="valid", output_reference=True)]
        test_hooks = [MetricsHook_mir(instrument_mappings, write_summaries=True, print_detailed=True, split="test"), BatchOutputWriterHook_mir(output_reference=True)]
        if args.save_salience:
            test_hooks.append(SaveSaliencesHook())
        print("preparing datasets...")
        train_dataset, test_datasets, validation_datasets = common.prepare_datasets(
            args.datasets, args, preload_fn, dataset_transform, dataset_transform_train, 
            small_hooks_mf0=small_hooks, valid_hooks=valid_hooks, test_hooks=test_hooks)
        print("done preparing datasets")

        all_notes = train_dataset.all_notes()
        notes_count = np.zeros((args.note_range,))
        for note_frame in all_notes:
            for note in note_frame:
                notes_count[int(note)] += 1
        
        class_priors = notes_count / np.sum(notes_count)
        mean_prior = 1 / args.note_range
        class_weights = mean_prior / class_priors * (1 - class_priors) / (1 - mean_prior)
        class_weights = class_weights ** 0.3
        print("weights", class_weights)

        # if not args.voicing:
        #     for vd in validation_datasets:
        #         if not vd.name.startswith("small_"):
        #             vd.hooks.append(AdjustVoicingHook())

        network.construct(args, create_model, train_dataset.dataset.output_types, train_dataset.dataset.output_shapes, spectrogram_info=spectrogram_info, class_weights=class_weights)

    return network, train_dataset, validation_datasets, test_datasets


if __name__ == "__main__":
    common.main(sys.argv[1:], construct, parse_args)
