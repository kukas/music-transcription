import numpy as np
import tensorflow as tf

import datasets
from .network import Network, safe_div

class NetworkMelody(Network):
    def _process_estimations(self, fetched_values):
        estimations = {}
        times_per_uid = fetched_values[self.times]
        est_notes_per_uid = fetched_values[self.est_notes]

        for uid in est_notes_per_uid.keys():
            est_time = times_per_uid[uid]
            est_note = est_notes_per_uid[uid]
            est_freq = datasets.common.midi_to_hz_safe(est_note)
            estimations[uid] = (est_time, est_note, est_freq)

        return estimations

    def _summaries(self, args):
        # batch metrics
        with tf.name_scope("metrics"):
            ref_notes = tf.cast(self.annotations[:, :, 0], tf.float32)
            est_diff = tf.abs(ref_notes-tf.abs(self.est_notes))

            voiced_ref = tf.greater(self.annotations[:, :, 0], 0)
            voiced_est = tf.greater(self.est_notes, 0)

            voiced_frames = tf.logical_and(voiced_ref, tf.not_equal(self.est_notes, 0))
            n_ref_sum = tf.cast(tf.count_nonzero(voiced_ref), tf.float64)

            correct = tf.less(est_diff, 0.5)
            correct_voiced_sum = tf.cast(tf.count_nonzero(tf.logical_and(correct, voiced_frames)), tf.float64)
            rpa = safe_div(correct_voiced_sum, n_ref_sum)
            rpa_sum = tf.Variable(0., dtype=tf.float64)
            self.train_metric = self.raw_pitch_accuracy = rpa_sum / self.train_stats_every
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
