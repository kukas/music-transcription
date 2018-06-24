from math import floor, ceil
import numpy as np

import tensorflow as tf

import mir_eval
mir_eval.multipitch.MIN_FREQ = 1

import visualization as vis

import datetime
import time

import inspect
import os

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, create_model, create_summaries = None):
        self.logdir = args["logdir"]
        self.note_range = args["note_range"]
        self.annotations_per_window = args["annotations_per_window"]
        self.window_size = args["window_size"]

        # save the model function for easier reproducibility
        self.save_model_fnc(create_model)

        with self.session.graph.as_default():

            # Inputs
            self.window = tf.placeholder(tf.float32, [None, self.window_size], name="window")
            self.annotations = tf.placeholder(tf.int32, [None, self.annotations_per_window, None], name="annotations")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            self.ref_notes = tf.reduce_sum(tf.one_hot(self.annotations, self.note_range), axis=2)
            # annotations are padded with zeros, so we manually remove this additional note
            dims = tf.shape(self.ref_notes)
            self.ref_notes = tf.concat([tf.zeros([dims[0], dims[1], 1]), self.ref_notes[:,:,1:]], axis=2)
            
            # create_model has to provide these values:
            self.est_notes = None
            self.loss = None
            self.training = None

            create_model(self, args)

            assert self.est_notes != None
            assert self.loss != None
            assert self.training != None

            # batch metrics
            with tf.name_scope("metrics"):
                ref_notes_b = tf.cast(self.ref_notes, tf.bool)
                est_notes_b = tf.cast(self.est_notes, tf.bool)
                true_positive_sum = tf.count_nonzero(tf.logical_and(ref_notes_b, est_notes_b))
                n_ref_sum = tf.count_nonzero(ref_notes_b)
                n_est_sum = tf.count_nonzero(est_notes_b)

                def safe_div(numerator, denominator):
                    return tf.cond(denominator > 0, lambda: numerator/denominator, lambda: tf.constant(0, dtype=tf.float64))

                self.precision = safe_div(true_positive_sum, n_est_sum)
                self.recall = safe_div(true_positive_sum, n_ref_sum)
                acc_denom = n_est_sum + n_ref_sum - true_positive_sum
                self.accuracy = safe_div(true_positive_sum, acc_denom)
                # self.precision = tf.Print(self.precision, [self.annotations, self.ref_notes, true_positive_sum, n_ref_sum, n_est_sum, ref_notes_b, est_notes_b], summarize=10000)

            # summaries
            summary_writer = tf.contrib.summary.create_file_writer(self.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(200):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.loss),
                                           tf.contrib.summary.scalar("train/precision", self.precision),
                                           tf.contrib.summary.scalar("train/recall", self.recall),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy),]

            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.given_loss = tf.placeholder(tf.float32, [], name="given_loss")
                self.given_precision = tf.placeholder(tf.float32, [], name="given_precision")
                self.given_recall = tf.placeholder(tf.float32, [], name="given_recall")
                self.given_accuracy = tf.placeholder(tf.float32, [], name="given_accuracy")
                self.given_e_sub = tf.placeholder(tf.float32, [], name="given_e_sub")
                self.given_e_miss = tf.placeholder(tf.float32, [], name="given_e_miss")
                self.given_e_fa = tf.placeholder(tf.float32, [], name="given_e_fa")
                self.given_e_tot = tf.placeholder(tf.float32, [], name="given_e_tot")
                self.given_precision_chroma = tf.placeholder(tf.float32, [], name="given_precision_chroma")
                self.given_recall_chroma = tf.placeholder(tf.float32, [], name="given_recall_chroma")
                self.given_accuracy_chroma = tf.placeholder(tf.float32, [], name="given_accuracy_chroma")
                self.given_e_sub_chroma = tf.placeholder(tf.float32, [], name="given_e_sub_chroma")
                self.given_e_miss_chroma = tf.placeholder(tf.float32, [], name="given_e_miss_chroma")
                self.given_e_fa_chroma = tf.placeholder(tf.float32, [], name="given_e_fa_chroma")
                self.given_e_tot_chroma = tf.placeholder(tf.float32, [], name="given_e_tot_chroma")

                self.image1 = tf.placeholder(tf.uint8, [None, None, 4], name="image1")
                image1 = tf.expand_dims(self.image1, 0)

                self.summaries["test_small"] = [tf.contrib.summary.image("test_small/image1", image1),
                                                tf.contrib.summary.scalar("test_small/loss", self.given_loss),
                                                tf.contrib.summary.scalar("test_small/precision", self.given_precision),
                                                tf.contrib.summary.scalar("test_small/recall", self.given_recall),
                                                tf.contrib.summary.scalar("test_small/accuracy", self.given_accuracy),]
                
                self.summaries["test"] = [tf.contrib.summary.scalar("test/loss", self.given_loss),
                                          tf.contrib.summary.scalar("test/precision", self.given_precision),
                                          tf.contrib.summary.scalar("test/recall", self.given_recall),
                                          tf.contrib.summary.scalar("test/accuracy", self.given_accuracy),
                                          tf.contrib.summary.scalar("test/e_sub", self.given_e_sub),
                                          tf.contrib.summary.scalar("test/e_miss", self.given_e_miss),
                                          tf.contrib.summary.scalar("test/e_fa", self.given_e_fa),
                                          tf.contrib.summary.scalar("test/e_tot", self.given_e_tot),
                                          tf.contrib.summary.scalar("test/precision_chroma", self.given_precision_chroma),
                                          tf.contrib.summary.scalar("test/recall_chroma", self.given_recall_chroma),
                                          tf.contrib.summary.scalar("test/accuracy_chroma", self.given_accuracy_chroma),
                                          tf.contrib.summary.scalar("test/e_sub_chroma", self.given_e_sub_chroma),
                                          tf.contrib.summary.scalar("test/e_miss_chroma", self.given_e_miss_chroma),
                                          tf.contrib.summary.scalar("test/e_fa_chroma", self.given_e_fa_chroma),
                                          tf.contrib.summary.scalar("test/e_tot_chroma", self.given_e_tot_chroma),
                                          # tf.contrib.summary.image('test/image1', self.image1),
                                          # tf.contrib.summary.image('test/estimation_detail', image_tensor_detail),
                                         ]

            if create_summaries:
                create_summaries(self, args, summary_writer)
            
            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def _unpack_one(self, batch, key):
        return [b[key] for b in batch]

    def _unpack_batch(self, batch):
        return self._unpack_one(batch, "audio"), self._unpack_one(batch, "annotation")

    def train_batch(self, batch):
        windows, annotations = self._unpack_batch(batch)

        out = self.session.run([self.accuracy, self.loss, self.training, self.summaries["train"]],
                         {self.window: windows, self.annotations: annotations, self.is_training: True})
        return out[0], out[1]
    
    def train(self, train_dataset, test_dataset, small_test_dataset, args):
        train_dataset.reset()
        batch_size = args["batch_size"]
        b = 0
        timer = time.time()
        for i in range(args["epochs"]):
            print("=== epoch", i+1, "===")
            while not train_dataset.epoch_finished():
                batch = train_dataset.next_batch(batch_size)
                accuracy, loss = self.train_batch(batch)

                b += 1
                if b % 200 == 0:
                    print(".", end="")
                if b % 1000 == 0:
                    print("b {0}; t {1:.2f}; acc {2:.2f}; loss {3:.2f}".format(b, time.time() - timer, accuracy, loss))
                    timer = time.time()
                if b % 3000 == 0:
                    self.evaluate(test_dataset, batch_size)
                    self.evaluate(small_test_dataset, batch_size, visual_output=True)
                    # reference, estimation, loss = network.predict(small_test_dataset, 2)
                    # fig, ax = vis.draw_notes(reference, estimation, ".")
                    # plt.show()
                    print("epoch {}, batch {}, t {:.2f}, accuracy: {:.2f}".format(i+1, b, time.time() - timer, accuracy))
                    timer = time.time()
                if b % 10000 == 0 and b >= 10000:
                    self.save()
                    print("saving, t {:.2f}".format(time.time()-timer))
                    timer = time.time()
        print("=== done ===")
    
    def evaluate(self, dataset, batch_size, visual_output=False):
        reference, estimation, loss = self.predict(dataset, batch_size)

        ref = np.array([mir_eval.util.midi_to_hz(np.array(notes)) for notes in reference])
        est = np.array([mir_eval.util.midi_to_hz(np.array(notes)) for notes in estimation])
        t = np.arange(0, len(ref))*0.01

        metrics = mir_eval.multipitch.metrics(t, ref, t, est)
        # unpack metrics
        (given_precision,given_recall,given_accuracy,
        given_e_sub,given_e_miss,given_e_fa,given_e_tot,
        given_precision_c,given_recall_c,given_accuracy_c,
        given_e_sub_c,given_e_miss_c,given_e_fa_c,given_e_tot_c) = metrics

        # write evaluation metrics to tf summary
        if visual_output:
            fig = vis.draw_notes(reference, estimation)
            image1 = vis.fig2data(fig)
            self.session.run(self.summaries["test_small"], {
                self.image1: image1,
                self.given_loss: loss,
                self.given_precision: given_precision,
                self.given_recall: given_recall,
                self.given_accuracy: given_accuracy,
            })
        else:
            self.session.run(self.summaries["test"], {
                self.given_loss: loss,
                # mir_eval summary
                self.given_precision: given_precision,
                self.given_recall: given_recall,
                self.given_accuracy: given_accuracy,
                self.given_e_sub: given_e_sub,
                self.given_e_miss: given_e_miss,
                self.given_e_fa: given_e_fa,
                self.given_e_tot: given_e_tot,
                self.given_precision_chroma: given_precision_c,
                self.given_recall_chroma: given_recall_c,
                self.given_accuracy_chroma: given_accuracy_c,
                self.given_e_sub_chroma: given_e_sub_c,
                self.given_e_miss_chroma: given_e_miss_c,
                self.given_e_fa_chroma: given_e_fa_c,
                self.given_e_tot_chroma: given_e_tot_c,
            })

        return given_accuracy
    
    def predict(self, dataset, batch_size):
        dataset.reset()
        estimation = []
        reference = []
        loss = 0
        c = 0
        while not dataset.epoch_finished():
            batch = dataset.next_batch(batch_size)
            batch_windows, batch_annotations = self._unpack_batch(batch)
            batch_annotations_ragged = self._unpack_one(batch, "annotation_ragged")

            batch_est_notes, batch_loss = self.session.run([self.est_notes, self.loss],
               {self.window: batch_windows, self.annotations: batch_annotations, self.is_training: False})

            est_annotation = [[i for i, v in enumerate(est_notes_frame) if v == 1] for est_notes in batch_est_notes for est_notes_frame in est_notes]
            ref_annotation = [est_notes_frame for est_notes in batch_annotations_ragged for est_notes_frame in est_notes]

            estimation += est_annotation
            reference += ref_annotation

            loss += batch_loss * len(batch_windows)
            c += len(batch_windows)
        loss /= c

        assert len(reference) == len(estimation)

        return reference, estimation, loss
    
    def save_model_fnc(self, create_model):
        plaintext = inspect.getsource(create_model)
        if not os.path.exists(self.logdir): os.mkdir(self.logdir)
        out = open(self.logdir+"/model_fnc.py", "a")
        out.write(plaintext)

    def save(self):
        save_path = self.saver.save(self.session, self.logdir+"/model.ckpt")
        print("Model saved in path:", save_path)
    def restore(self):
        self.saver.restore(self.session, self.logdir+"/model.ckpt")
        print("Model restored from path:", self.logdir)