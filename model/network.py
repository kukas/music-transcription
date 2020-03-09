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

try:
    from mem_top import mem_top
except ImportError:
    print("mem_top package not found")


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
        self.min_note = args.min_note

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
            
            self.train_stats_every = 1000
            self.train_metric = None
            self.average_loss = None
            self.averages_clear_op = None
            self.averages_inc_op = None
            self.summaries = None

            self._summaries(args)

            assert self.train_metric is not None
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
    
    def _print_train_summary(self, b, elapsed, train_metric, loss):
        print("b {0}; t {1:.2f}; RPA {2:.2f}; loss {3:.4f}".format(b, elapsed, train_metric, loss))

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
                            fetches = [self.train_metric, self.average_loss, self.training, self.summaries, self.global_step, self.averages_clear_op]

                            if self.args.full_trace:
                                print("Running with trace_level=FULL_TRACE")
                                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                run_metadata = tf.RunMetadata()

                                values = self.session.run(fetches, feed_dict, options=run_options, run_metadata=run_metadata)
                                self.summary_writer.add_run_metadata(run_metadata, 'run_metadata_step{}'.format(b))
                            else:
                                values = self.session.run(fetches, feed_dict)

                            train_metric, loss, _, summary, step, _ = values
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
                    if self.args.stop_if_too_slow is not None and elapsed > self.args.stop_if_too_slow:
                        self.save(self.args.checkpoint)
                        return

                    self._print_train_summary(b, elapsed, train_metric, loss)
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
                if self.args.iterations is not None and b >= self.args.iterations:
                    self.args.epochs = 0 # breaks the outer loop
                    break

        print("=== done ===")

    def _predict_handle(self, handle, additional_fetches=[]):
        # TODO take voicing probabilities back
        fetches = [self.audio_uid, self.times, self.note_probabilities, self.est_notes]
        # de-dup fetches while preserving order
        for f in additional_fetches:
            if f not in fetches:
                fetches.append(f)

        feed_dict = {
            self.is_training: False,
            self.handle: handle
        }

        all_values = [None] * len(fetches)

        last_uid = None
        first_run = True
        timer_session_run = 0
        timer_postprocess = 0
        while True:
            timer_session_run -= time.time()
            try:
                fetched_values = self.session.run(fetches, feed_dict)
            except tf.errors.OutOfRangeError:
                timer_session_run += time.time()
                break
            timer_session_run += time.time()
            
            timer_postprocess -= time.time()

            uids = fetched_values[0] # shape = (batch_size)
            uids = np.array(list(map(lambda uid: uid.decode("utf-8"), uids)))
            times = fetched_values[1] # shape = (batch_size, annotations_per_window)
            mask = times >= 0

            # print("uids.shape", uids.shape)
            # print("times.shape", times.shape)
            # print("mask.shape", mask.shape)
            
            if first_run:
                for i, fetched_value in enumerate(fetched_values):
                    if fetched_value.shape == ():
                        all_values[i] = []
                    else:
                        all_values[i] = defaultdict(list)
                first_run = False
            
            for all_values_bin, fetched_value in zip(all_values, fetched_values):
                # fetched_value might have shape (), (batch_size), (batch_size, annotations_per_window), (batch_size, annotations_per_window, ...) or other shape
                # print(fetched_value.shape)

                # if fetched_value has shape (batch_size, ...)
                if fetched_value.shape != () and fetched_value.shape[0] == len(uids):
                    # print("batched values", fetched_value.shape, len(fetched_value.shape))
                    # iterate through the tensors in one batch
                    for uid, window_mask, window in zip(uids, mask, fetched_value):
                        # if there is only one value per batch per fetch
                        if len(fetched_value.shape) == 1:
                            all_values_bin[uid].append(window)
                        # if there is a window of values per batch per fetch, 
                        # we want to mask those according to window mask
                        if len(fetched_value.shape) >= 2:
                            all_values_bin[uid].append(window[window_mask])
                else:
                    # if the number of values in fetch doesn't match the number of batches
                    # print("non batched values", fetched_value.shape)
                    all_values_bin.append(fetched_value)

            if last_uid is None or last_uid != uids[0]:
                print(".", end="")
                last_uid = uids[0]
            
            timer_postprocess += time.time()
        
        print()
        print("timer_session_run {:.2f}s".format(timer_session_run))

        timer_postprocess -= time.time()

        structured_fetched_values = dict(zip(fetches, all_values))
        for fetch_key, fetched_values in structured_fetched_values.items():
            if isinstance(fetched_values, dict):
                # if fetched_values is a dict, it contains lists of values divided to separate uids
                for uid, windows in fetched_values.items():
                    # for each uid we concatenate all the tensors -> we get one continuous matrix of values
                    # instead of values divided to batches and windows
                    if len(windows) > 0 and hasattr(windows[0], "shape"):
                        fetched_values[uid] = np.concatenate(windows)
            elif isinstance(fetched_values, list):
                structured_fetched_values[fetch_key] = np.array(fetched_values)

        timer_postprocess += time.time()

        print("timer_postprocess {:.2f}s".format(timer_postprocess))
        # the resulting dict contains either dicts indexed by uid of the datapoint
        # or list of values that are not assigned to any datapoint uid
        return structured_fetched_values
    
    def _process_estimations(self, fetched_values):
        raise NotImplementedError

    def _evaluate_handle(self, vd, handle):
        if self.args.debug_memory_leaks:
            print(mem_top())

        additional_fetches = []

        for hook in vd.hooks:
            hook_fetches = hook.before_predict(self, vd)
            if hook_fetches is not None:
                additional_fetches += hook_fetches

        fetched_values = self._predict_handle(handle, additional_fetches)

        timer = time.time()
        estimations = self._process_estimations(fetched_values)
        print("_process_estimations() {:.2f}s".format(time.time()-timer))

        for hook in vd.hooks:
            hook.after_predict(self, vd, estimations, fetched_values)

        for uid, (est_time, est_notes, est_freqs) in estimations.items():
            aa = vd.dataset.get_annotated_audio_by_uid(uid)

            for hook in vd.hooks:
                hook.every_aa(self, vd, aa, est_time, est_notes, est_freqs)

        for hook in vd.hooks:
            hook.after_run(self, vd, fetched_values)

        if self.args.debug_memory_leaks:
            print(mem_top())

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

