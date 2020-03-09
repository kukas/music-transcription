import numpy as np
import tensorflow as tf

import datasets
from .network import Network, safe_div
import time
from numba import jit

@jit(nopython=True)
def _process_fetched_values(probs, bins_per_semitone, min_note):
    notes = []
    freqs = []
    for i in range(len(probs)):
        frame = probs[i]
        notes_in_frame = np.array([i/bins_per_semitone + min_note for i, prob in enumerate(frame) if prob > 0.5])
        notes.append(notes_in_frame)
        freqs.append(440.0 * (2.0 ** ((notes_in_frame - 69.0)/12.0)))
    
    return notes, freqs

class NetworkMultif0(Network):
    def _process_estimations(self, fetched_values):
        times_per_uid = fetched_values[self.times]
        note_probabilities_per_uid = fetched_values[self.note_probabilities]
        estimations = {}
        # !!! TODO: variable note probability threshold
        for uid, probs in note_probabilities_per_uid.items():
            est_notes, est_freqs = _process_fetched_values(probs, self.args.bins_per_semitone, self.min_note)
            time = times_per_uid[uid]
            estimations[uid] = (time, est_notes, est_freqs)
        
        return estimations

    def _summaries(self, args):
        # batch metrics
        with tf.name_scope("metrics"):
            ref_matrix = tf.cast(tf.reduce_sum(tf.one_hot(tf.cast(tf.math.round(self.annotations - self.min_note), tf.int32), self.note_range), -2), tf.bool)
            print("ref_matrix shape", ref_matrix.shape)
            est_matrix = tf.greater(self.note_probabilities, tf.constant(0.5))

            tf.summary.image("metrics/ref_matrix", tf.expand_dims(tf.cast(ref_matrix, tf.float32), -1), max_outputs=1)
            tf.summary.image("metrics/est_matrix", tf.expand_dims(tf.cast(est_matrix, tf.float32), -1), max_outputs=1)
            tf.summary.image("metrics/note_probabilities", tf.expand_dims(self.note_probabilities, -1), max_outputs=1)

            tp = tf.count_nonzero(tf.math.logical_and(ref_matrix, est_matrix), dtype=tf.float64)
            fp = tf.count_nonzero(tf.math.logical_and(tf.math.logical_not(ref_matrix), est_matrix), dtype=tf.float64)
            fn = tf.count_nonzero(tf.math.logical_and(ref_matrix, tf.math.logical_not(est_matrix)), dtype=tf.float64)

            p = safe_div(tp, tp + fp)
            p_sum = tf.Variable(0., dtype=tf.float64)
            self.precision = p_sum / self.train_stats_every

            r = safe_div(tp, tp + fn)
            r_sum = tf.Variable(0., dtype=tf.float64)
            self.recall = r_sum / self.train_stats_every

            f1 = safe_div(2 * p * r, p + r)
            f1_sum = tf.Variable(0., dtype=tf.float64)
            self.train_metric = self.f1 = f1_sum / self.train_stats_every

            tf.summary.scalar("train/precision", self.precision)
            tf.summary.scalar("train/recall", self.recall)
            tf.summary.scalar("train/f1", self.f1)

            loss_sum = tf.Variable(0.)
            self.average_loss = loss_sum / self.train_stats_every
            tf.summary.scalar("train/average_loss", self.average_loss)
            tf.summary.scalar("train/loss", self.loss)

            self.summaries = tf.summary.merge_all()
            with tf.control_dependencies([self.summaries]):
                inc_ops = []
                clear_ops = []
                for sum_var, var in [(loss_sum, self.loss), (p_sum, p), (r_sum, r), (f1_sum, f1)]:
                    inc_ops.append(tf.assign_add(sum_var, var))
                    clear_ops.append(tf.assign(sum_var, 0))
                self.averages_inc_op = tf.group(*inc_ops)
                self.averages_clear_op = tf.group(*clear_ops)


    def _print_train_summary(self, b, elapsed, train_metric, loss):
        print("b {0}; t {1:.2f}; F1 {2:.2f}; loss {3:.4f}".format(b, elapsed, train_metric, loss))