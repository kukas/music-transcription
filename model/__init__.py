from .hooks import *
import pandas
import evaluation
import datasets
import os
import inspect
import time
import visualization as vis
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
from collections import namedtuple
from math import floor, ceil
import numpy as np

import tensorflow as tf

import mir_eval
mir_eval.multipitch.MIN_FREQ = 1


VD = namedtuple("ValidationDataset", ["name", "dataset", "evaluate_every", "hooks"])


class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options,
                                                                       inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, create_model, output_types, output_shapes, create_summaries=None, dataset_preload_fn=None, dataset_transform=None):
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
        if "spectrogram_shape" in args:
            self.spectrogram_shape = args.spectrogram_shape

        self.dataset_preload_fn = dataset_preload_fn
        self.dataset_transform = dataset_transform

        with self.session.graph.as_default():
            # Inputs
            self.handle = tf.placeholder(tf.string, shape=[])
            self.iterator = tf.data.Iterator.from_string_handle(self.handle, output_types, output_shapes)

            self.window_int16, self.annotations, self.times, self.audio_uid = self.iterator.get_next()

            self.window = tf.cast(self.window_int16, tf.float32)/32768.0

            # if self.spectrogram_shape is not None:
            #     self.spectrogram = tf.placeholder(tf.float32, [None, self.spectrogram_shape[0], self.spectrogram_shape[1]], name="spectrogram")

            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            self.global_step = tf.train.create_global_step()

            # lowpriority TODO: použít booleanmask, abych pak nemusel odstraňovat ty nulové anotace
            # self.ref_notes_sparse = tf.reduce_sum(tf.one_hot(self.annotations, self.note_range), axis=2)
            # sometimes there are identical notes in annotations (played by two different instruments)
            # self.ref_notes_sparse = tf.cast(tf.greater(self.ref_notes_sparse, 0), tf.float32)
            # annotations are padded with zeros, so we manually remove this additional note
            # dims = tf.shape(self.ref_notes_sparse)
            # self.ref_notes_sparse = tf.concat([tf.zeros([dims[0], dims[1], 1]), self.ref_notes_sparse[:,:,1:]], axis=2)

            # create_model has to provide these values:
            self.note_probabilities = None
            self.est_notes = None
            self.loss = None
            self.training = None

            create_model(self, args)

            print("Total parameter count:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

            assert self.note_probabilities is not None and self.note_probabilities.shape.as_list() == [None, self.annotations_per_window, self.bin_count]
            assert self.est_notes is not None
            assert self.loss is not None
            assert self.training is not None

            self.saver = tf.train.Saver()

            self.summary_writer = tf.summary.FileWriter(self.logdir, graph=self.session.graph, flush_secs=30)

            self.accuracy = None
            self.summaries = None

            self._summaries(args)

            assert self.accuracy is not None
            assert self.summaries is not None

            # save the model function for easier reproducibility
            self.save_model_fn(create_model)
            self.save_hyperparams(args)

            if create_summaries:
                create_summaries(self, args)

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

            if os.path.exists(os.path.join(self.logdir, "model.ckpt.index")):
                self.restore()

    def _summaries(self, args):
        raise NotImplementedError()

    def train(self, train_dataset, epochs, validation_datasets, save_every_n_batches=20000):
        validation_iterators = []
        validation_handles = []
        with self.session.graph.as_default():
            train_iterator = train_dataset.dataset.make_initializable_iterator()
            train_iterator_handle = self.session.run(train_iterator.string_handle())

            for vd in validation_datasets:
                it = vd.dataset.dataset.make_initializable_iterator()
                validation_iterators.append(it)
                validation_handles.append(self.session.run(it.string_handle()))

        b = 0
        timer = time.time()
        for i in range(epochs):
            self.session.run(train_iterator.initializer)
            print("=== epoch", i+1, "===")
            while True:
                feed_dict = {
                    self.handle: train_iterator_handle,
                    self.is_training: True
                }

                try:
                    if b % 1000 == 0:
                        fetches = [self.accuracy, self.loss, self.training, self.summaries, self.global_step]
                        if self.args.full_trace:
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()

                            values = self.session.run(fetches, feed_dict, options=run_options, run_metadata=run_metadata)
                            self.summary_writer.add_run_metadata(run_metadata, 'run_metadata_step{}'.format(step), global_step=step)
                        else:
                            values = self.session.run(fetches, feed_dict)

                        accuracy, loss, _, summary, step = values
                    else:
                        self.session.run(self.training, feed_dict)
                except tf.errors.OutOfRangeError:
                    break

                b += 1

                if b % 200 == 0:
                    print(".", end="")

                if b % 1000 == 0:
                    self.summary_writer.add_summary(summary, step)
                    print("b {0}; t {1:.2f}; acc {2:.2f}; loss {3:.2f}".format(b, time.time() - timer, accuracy, loss))
                    timer = time.time()

                for vd, iterator, handle in zip(validation_datasets, validation_iterators, validation_handles):
                    if b % vd.evaluate_every == 0:
                        self.session.run(iterator.initializer)
                        self._evaluate_handle(vd, handle)
                        print("  time: {:.2f}".format(time.time() - timer))
                        timer = time.time()

                if b % save_every_n_batches == 0:
                    self.save()
                    print("saving, t {:.2f}".format(time.time()-timer))
                    timer = time.time()

        print("=== done ===")

    def _predict_handle(self, handle, additional_fetches=[]):
        feed_dict = {
            self.is_training: False,
            self.handle: handle
        }

        notes = defaultdict(list)
        times = defaultdict(list)
        additional = []
        while True:
            try:
                fetches = self.session.run([self.est_notes, self.times, self.audio_uid]+additional_fetches, feed_dict)
            except tf.errors.OutOfRangeError:
                break

            additional.append(fetches[3:])
            # iterates through the batch output
            for est_notes_window, times_window, uid in zip(*fetches[:3]):
                uid = uid.decode("utf-8")
                notes[uid].append(est_notes_window)
                times[uid].append(times_window)

        estimations = {}
        for uid in notes.keys():
            est_time = np.concatenate(times[uid])
            est_notes = np.concatenate(notes[uid])

            # if there is padding at the end of the estimation, cut it
            zeros = np.where(est_time == 0)[0]
            if len(zeros) > 1:
                est_time = est_time[:zeros[1]]
                est_notes = est_notes[:zeros[1]]

            est_freq = datasets.common.midi_to_hz_safe(est_notes)
            estimations[uid] = (est_time, est_freq)

        if additional_fetches:
            return estimations, additional

        return estimations, []

    def predict(self, dataset_iterator, name="predict"):
        predict_data = datasets.load_melody_dataset(name, dataset_iterator)
        with self.session.graph.as_default():
            predict_dataset = datasets.AADataset(predict_data, self.args, self.dataset_transform)
            iterator = predict_dataset.dataset.make_one_shot_iterator()
        handle = self.session.run(iterator.string_handle())

        return self._predict_handle(handle)

    def _evaluate_handle(self, dataset, handle, dataset_name=None, visual_output=None, print_detailed=False):
        raise NotImplementedError()

    def evaluate(self, dataset, **kwargs):
        with self.session.graph.as_default():
            iterator = dataset.dataset.make_one_shot_iterator()
        handle = self.session.run(iterator.string_handle())

        return self._evaluate_handle(dataset, handle, **kwargs)

    def save_model_fn(self, create_model):
        source = inspect.getsource(create_model)
        source = "\t"+source.replace("\n", "\t\n").replace("    ", "\t")
        text = tf.convert_to_tensor(source)
        self.summary_writer.add_summary(self.session.run(tf.summary.text("model_fn", text)))

        # if not os.path.exists(self.logdir): os.mkdir(self.logdir)
        # out = open(self.logdir+"/model_fnc.py", "a")
        # out.write(plaintext)

    def save_hyperparams(self, args):
        text = tf.convert_to_tensor([[str(k), str(v)] for k, v in vars(args).items()])
        self.summary_writer.add_summary(self.session.run(tf.summary.text("hyperparameters", text)))

    def save(self, name="model"):
        save_path = self.saver.save(self.session, self.logdir+"/"+name+".ckpt")
        print("Model saved in path:", save_path)

    def restore(self):
        self.saver.restore(self.session, self.logdir+"/model.ckpt")
        print("Model restored from path:", self.logdir)


class NetworkMelody(Network):
    def _summaries(self, args):
        # batch metrics
        with tf.name_scope("metrics"):
            ref_notes = tf.cast(self.annotations[:, 0], tf.float64)
            correct = tf.less(tf.abs(ref_notes-self.est_notes), 0.5)
            voiced = tf.greater(self.annotations[:, 0], 0)
            voiced_est = tf.greater(self.est_notes, 0)
            correct_voiced_sum = tf.count_nonzero(tf.logical_and(correct, voiced))
            n_ref_sum = tf.count_nonzero(voiced)
            n_est_sum = tf.count_nonzero(voiced_est)

            def safe_div(numerator, denominator):
                return tf.cond(denominator > 0, lambda: numerator/denominator, lambda: tf.constant(0, dtype=tf.float64))

            self.precision = safe_div(correct_voiced_sum, n_est_sum)
            self.recall = safe_div(correct_voiced_sum, n_ref_sum)
            acc_denom = n_est_sum + n_ref_sum - correct_voiced_sum
            self.accuracy = safe_div(correct_voiced_sum, acc_denom)

            tf.summary.scalar("train/loss", self.loss),
            tf.summary.scalar("train/precision", self.precision),
            tf.summary.scalar("train/recall", self.recall),
            tf.summary.scalar("train/accuracy", self.accuracy)

            self.summaries = tf.summary.merge_all()

    def _evaluate_handle(self, vd, handle):
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
