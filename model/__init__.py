from .hooks import *
import pandas
import evaluation
import datasets
import os
import sys
import inspect
import time
import visualization as vis
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
from collections import namedtuple
from math import floor, ceil
import numpy as np

from mem_top import mem_top

import tensorflow as tf

VD = namedtuple("ValidationDataset", ["name", "dataset", "evaluate_every", "hooks"])

def safe_div(numerator, denominator):
    return tf.where(tf.less(denominator, 1e-7), denominator, numerator/denominator)

class Network:
    def __init__(self, args, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed

        gpu_options = tf.GPUOptions(allow_growth=True)
        device_count = None
        if args.cpu:
            device_count = {'GPU': 0}
        config = tf.ConfigProto(gpu_options=gpu_options,
                                device_count=device_count,
                                inter_op_parallelism_threads=args.threads,
                                intra_op_parallelism_threads=args.threads)
        self.session = tf.Session(graph=graph, config=config)

    def construct(self, args, create_model, output_types, output_shapes, create_summaries=None, spectrogram_info=None):
        self.args = args
        self.logdir = args.logdir
        self.note_range = args.note_range

        self.samplerate = args.samplerate
        self.bins_per_semitone = args.bins_per_semitone
        self.annotations_per_window = args.annotations_per_window
        self.bin_count = self.note_range*self.bins_per_semitone
        self.frame_width = args.frame_width
        self.context_width = args.context_width
        self.window_size = self.annotations_per_window*self.frame_width + 2*self.context_width

        self.spectrogram_shape = None
        self.spectrogram_hop_size = None
        if spectrogram_info is not None:
            channels, freq_bins, self.spectrogram_hop_size, self.spectrogram_fmin = spectrogram_info
            
            width = self.annotations_per_window + 2*self.context_width/self.spectrogram_hop_size
            self.spectrogram_shape = [channels, freq_bins, width]

        with self.session.graph.as_default():
            # Inputs
            self.handle = tf.placeholder(tf.string, shape=[])
            self.iterator = tf.data.Iterator.from_string_handle(self.handle, output_types, output_shapes)

            with tf.name_scope("input"):

                self.window_int16, self.spectrogram_uint16, self.annotations, self.times, self.audio_uid = self.iterator.get_next()

                self.window = tf.cast(self.window_int16, tf.float32)/32768.0
                self.spectrogram = tf.cast(self.spectrogram_uint16, tf.float32)/65536.0

                if self.spectrogram_shape is not None:
                    self.spectrogram.set_shape([None]+self.spectrogram_shape)
                    self.spectrogram = tf.transpose(self.spectrogram, [0, 3, 2, 1])

                self.is_training = tf.placeholder(tf.bool, [], name="is_training")

                self.global_step = tf.train.create_global_step()

                batch_size = tf.shape(self.annotations)[0]
                self.note_bins = tf.range(0, self.note_range, 1/self.bins_per_semitone, dtype=tf.float32)
                self.note_bins = tf.reshape(tf.tile(self.note_bins, [batch_size * self.annotations_per_window]), [batch_size, self.annotations_per_window, self.bin_count])

            # lowpriority TODO: použít booleanmask, abych pak nemusel odstraňovat ty nulové anotace
            # self.ref_notes_sparse = tf.reduce_sum(tf.one_hot(self.annotations, self.note_range), axis=2)
            # sometimes there are identical notes in annotations (played by two different instruments)
            # self.ref_notes_sparse = tf.cast(tf.greater(self.ref_notes_sparse, 0), tf.float32)
            # annotations are padded with zeros, so we manually remove this additional note
            # dims = tf.shape(self.ref_notes_sparse)
            # self.ref_notes_sparse = tf.concat([tf.zeros([dims[0], dims[1], 1]), self.ref_notes_sparse[:,:,1:]], axis=2)

            # create_model has to provide these values:
            self.note_logits = None
            self.note_probabilities = None
            # or
            self.voicing_threshold = tf.constant(1.0)
            self.est_notes = None
            self.voicing_logits = None
            self.loss = None
            self.training = None

            create_model(self, args)

            self.trainable_parameter_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            print("Total parameter count:", self.trainable_parameter_count)

            assert self.note_probabilities is not None and self.note_probabilities.shape.as_list() == [None, self.annotations_per_window, self.bin_count]
            assert self.est_notes is not None
            assert self.loss is not None
            assert self.training is not None

            self.saver = tf.train.Saver(max_to_keep=args.saver_max_to_keep)
            self.saver_best = tf.train.Saver(max_to_keep=0)

            self.summary_writer = tf.summary.FileWriter(self.logdir, graph=self.session.graph, flush_secs=30)

            self.raw_pitch_accuracy = None
            self.summaries = None

            self._summaries(args)

            assert self.raw_pitch_accuracy is not None
            assert self.summaries is not None

            if create_summaries:
                create_summaries(self, args)

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

            checkpoint_path = os.path.join(self.logdir, args.checkpoint)
            if os.path.exists(checkpoint_path+".index"):
                self.restored = True
                self.restore(checkpoint_path)
            else:
                self.restored = False
                # save the model function for easier reproducibility
                self.save_model_fn(create_model)
                self.save_hyperparams(args)
            
    def _summaries(self, args):
        raise NotImplementedError()

    def train(self, train_dataset, validation_datasets, save_every_n_batches=20000):
        validation_iterators = []
        validation_handles = []
        with self.session.graph.as_default():
            train_iterator = train_dataset.dataset.make_initializable_iterator()
            train_iterator_handle = self.session.run(train_iterator.string_handle())

            for vd in validation_datasets:
                it = vd.dataset.dataset.make_initializable_iterator()
                validation_iterators.append(it)
                validation_handles.append(self.session.run(it.string_handle()))
        
        # self.session.graph.finalize()

        b = -1
        step = tf.train.global_step(self.session, self.global_step)
        rewind = self.args.rewind

        timer = time.time()
        epoch = 0
        while True:
            epoch += 1
            if self.args.epochs is not None and epoch > self.args.epochs:
                break
            self.session.run(train_iterator.initializer)
            print("=== epoch", epoch, "===")
            while True:
                feed_dict = {
                    self.handle: train_iterator_handle,
                    self.is_training: True
                }

                try:
                    if rewind and b < step-1:
                        self.session.run(self.times, feed_dict)
                    else:
                        rewind = False

                        if (b+1) % self.train_stats_every == 0:
                            fetches = [self.raw_pitch_accuracy, self.average_loss, self.training, self.summaries, self.global_step, self.averages_clear_op]

                            if self.args.full_trace:
                                print("Running with trace_level=FULL_TRACE")
                                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                run_metadata = tf.RunMetadata()

                                values = self.session.run(fetches, feed_dict, options=run_options, run_metadata=run_metadata)
                                self.summary_writer.add_run_metadata(run_metadata, 'run_metadata_step{}'.format(b))
                            else:
                                values = self.session.run(fetches, feed_dict)

                            raw_pitch_accuracy, loss, _, summary, step, _ = values
                        else:
                            _, step, _ = self.session.run([self.training, self.global_step, self.averages_inc_op], feed_dict)

                except tf.errors.OutOfRangeError:
                    break
                
                b += 1

                if b % 200 == 0 and b != 0:
                    print(".", end="")

                if rewind:
                    continue

                flush = False
                if b % self.train_stats_every == 0 and b != 0:
                    self.summary_writer.add_summary(summary, step)
                    flush = True

                    if b != step:
                        print("step {};".format(step), end=" ")
                    
                    elapsed = time.time() - timer

                    print("b {0}; t {1:.2f}; RPA {2:.2f}; loss {3:.4f}".format(b, elapsed, raw_pitch_accuracy, loss))
                    timer = time.time()

                for vd, iterator, handle in zip(validation_datasets, validation_iterators, validation_handles):
                    if b % vd.evaluate_every == 0 and b != 0:
                        self.session.run(iterator.initializer)
                        self._evaluate_handle(vd, handle)
                        flush = True

                        print("  time: {:.2f}".format(time.time() - timer))
                        timer = time.time()

                if b % save_every_n_batches == 0 and b != 0:
                    self.save(self.args.checkpoint)

                    print("saving, t {:.2f}".format(time.time()-timer))
                    timer = time.time()
                
                if flush:
                    self.summary_writer.flush()

        print("=== done ===")

    def _predict_handle(self, handle, additional_fetches=[]):
        # de-dup additional_fetches
        additional_fetches = list(set(additional_fetches))
        fetches = [self.audio_uid, self.times, self.est_notes]+additional_fetches

        feed_dict = {
            self.is_training: False,
            self.handle: handle
        }

        all_values = [None] * len(fetches)

        last_uid = None
        first_run = True
        while True:
            try:
                fetched_values = self.session.run(fetches, feed_dict)
            except tf.errors.OutOfRangeError:
                break
            
            uids = fetched_values[0]
            uids = list(map(lambda uid: uid.decode("utf-8"), uids))
            times = fetched_values[1]
            mask = times >= 0
            
            if first_run:
                for i, value in enumerate(fetched_values):
                    if value.shape == ():
                        all_values[i] = []
                    else:
                        all_values[i] = defaultdict(list)
                first_run = False
            
            for all_values_bin, value in zip(all_values, fetched_values):
                if value.shape != () and value.shape[0] == len(uids):
                    for uid, window_mask, window in zip(uids, mask, value):
                        if len(value.shape) == 1:
                            all_values_bin[uid].append(window)
                        if len(value.shape) >= 2:
                            all_values_bin[uid].append(window[window_mask])
                else:
                    all_values_bin.append(value)

            if last_uid is None or last_uid != uids[0]:
                print(".", end="")
                last_uid = uids[0]
        print()

        estimations = {}
        times = all_values[1]
        notes = all_values[2]
        for uid in notes.keys():
            est_time = np.concatenate(times[uid])
            est_notes = np.concatenate(notes[uid])

            for all_values_bin in all_values[3:]:
                if not isinstance(all_values_bin, list):
                    all_values_bin[uid] = np.concatenate(all_values_bin[uid])

            est_freq = datasets.common.midi_to_hz_safe(est_notes)
            estimations[uid] = (est_time, est_freq)

        return estimations, dict(zip(fetches, all_values))

    def _evaluate_handle(self, vd, handle):
        raise NotImplementedError()

    def evaluate(self, vd):
        with self.session.graph.as_default():
            iterator = vd.dataset.dataset.make_one_shot_iterator()
        handle = self.session.run(iterator.string_handle())

        return self._evaluate_handle(vd, handle)

    def save_model_fn(self, create_model):
        source = inspect.getsource(create_model)
        source = "\t"+source.replace("\n", "\n\t").replace("    ", "\t")
        text = tf.convert_to_tensor(source)
        self.summary_writer.add_summary(self.session.run(tf.summary.text("model_fn", text)))

        # if not os.path.exists(self.logdir): os.mkdir(self.logdir)
        # out = open(self.logdir+"/model_fnc.py", "a")
        # out.write(plaintext)

    def save_hyperparams(self, args):
        text = tf.convert_to_tensor([[str(k), str(v)] for k, v in vars(args).items()])
        self.summary_writer.add_summary(self.session.run(tf.summary.text("hyperparameters", text)))

        text = tf.convert_to_tensor([["trainable_parameter_count", str(self.trainable_parameter_count)],
                                     ["window_size", str(self.window_size)],
                                     ["bin_count", str(self.bin_count)]])
        self.summary_writer.add_summary(self.session.run(tf.summary.text("model_info", text)))

        text = tf.convert_to_tensor(" ".join(sys.argv[:]))
        self.summary_writer.add_summary(self.session.run(tf.summary.text("run_command", text)))

    def save(self, name, saver=None):
        if saver is None:
            saver = self.saver
        save_path = os.path.join(self.logdir, name)
        save_path = saver.save(self.session, save_path)
        print("Model saved in path:", save_path)

    def restore(self, restore_path):
        self.saver.restore(self.session, restore_path)
        print("Model restored from path:", restore_path)


class NetworkMelody(Network):
    def _summaries(self, args):
        # batch metrics
        with tf.name_scope("metrics"):
            ref_notes = tf.cast(self.annotations[:, :, 0], tf.float32)
            est_diff = tf.abs(ref_notes-tf.abs(self.est_notes))

            voiced_ref = tf.greater(self.annotations[:, :, 0], 0)
            voiced_est = tf.greater(self.est_notes, 0)

            voiced_frames = tf.logical_and(voiced_ref, voiced_est)
            n_ref_sum = tf.cast(tf.count_nonzero(voiced_ref), tf.float64)

            correct = tf.less(est_diff, 0.5)
            correct_voiced_sum = tf.cast(tf.count_nonzero(tf.logical_and(correct, voiced_frames)), tf.float64)
            rpa = safe_div(correct_voiced_sum, n_ref_sum)
            rpa_sum = tf.Variable(0., dtype=tf.float64)
            self.raw_pitch_accuracy = rpa_sum / self.train_stats_every
            tf.summary.scalar("train/raw_pitch_accuracy", self.raw_pitch_accuracy)

            octave = 12 * tf.floor(est_diff/12.0 + 0.5)
            correct_chroma = tf.less(est_diff-octave, 0.5)
            correct_chroma_voiced_sum = tf.cast(tf.count_nonzero(tf.logical_and(correct_chroma, voiced_frames)), tf.float64)
            rca = safe_div(correct_chroma_voiced_sum, n_ref_sum)
            rca_sum = tf.Variable(0., dtype=tf.float64)
            self.raw_chroma_accuracy = rca_sum / self.train_stats_every
            tf.summary.scalar("train/raw_chroma_accuracy", self.raw_chroma_accuracy)

            unvoiced_ref = tf.logical_not(voiced_ref)
            unvoiced_est = tf.logical_not(voiced_est)
            correct_unvoiced_sum = tf.cast(tf.count_nonzero(tf.logical_and(unvoiced_ref, unvoiced_est)), tf.float64)
            n_all_sum = tf.cast(tf.count_nonzero(unvoiced_ref), tf.float64) + n_ref_sum
            acc = safe_div(correct_voiced_sum+correct_unvoiced_sum, n_all_sum)
            acc_sum = tf.Variable(0., dtype=tf.float64)
            self.accuracy = acc_sum / self.train_stats_every
            tf.summary.scalar("train/overall_accuracy", self.accuracy)

            loss_sum = tf.Variable(0.)
            self.average_loss = loss_sum / self.train_stats_every
            tf.summary.scalar("train/average_loss", self.average_loss)
            tf.summary.scalar("train/loss", self.loss)

            self.summaries = tf.summary.merge_all()
            with tf.control_dependencies([self.summaries]):
                inc_ops = []
                clear_ops = []
                for sum_var, var in [(loss_sum, self.loss), (rpa_sum, rpa), (rca_sum, rca), (acc_sum, acc)]:
                    inc_ops.append(tf.assign_add(sum_var, var))
                    clear_ops.append(tf.assign(sum_var, 0))
                self.averages_inc_op = tf.group(*inc_ops)
                self.averages_clear_op = tf.group(*clear_ops)

    def _evaluate_handle(self, vd, handle):
        if self.args.debug_memory_leaks:
            print(mem_top())

        additional_fetches = []

        for hook in vd.hooks:
            hook_fetches = hook.before_run(self, vd)
            if hook_fetches is not None:
                additional_fetches += hook_fetches

        estimations, additional = self._predict_handle(handle, additional_fetches)

        for uid, (est_time, est_freq) in estimations.items():
            aa = vd.dataset.get_annotated_audio_by_uid(uid)

            for hook in vd.hooks:
                hook.every_aa(self, vd, aa, est_time, est_freq)

        for hook in vd.hooks:
            hook.after_run(self, vd, additional)

        if self.args.debug_memory_leaks:
            print(mem_top())
