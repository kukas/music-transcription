from numba import jit
from .network_multiir import _process_fetched_values

import tensorflow as tf
import pandas
import mir_eval
import evaluation
import datasets
import numpy as np
import visualization as vis
import matplotlib.pyplot as plt
import matplotlib
import os
import csv
import time
import sklearn
mir_eval.multipitch.MIN_FREQ = 1


def simplify_name(name):
    return name.lower().replace(" ", "_")

def add_fig(fig, summary_writer, tag, global_step=0):
    img_summary = vis.fig2summary(fig)
    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, image=img_summary)]), global_step)
    plt.cla()
    plt.clf()
    plt.close('all')

class EvaluationHook:
    def before_predict(self, ctx, vd):
        pass

    def after_predict(self, ctx, vd, estimations, additional):
        pass

    def every_aa(self, ctx, vd, aa, est):
        pass


    def after_run(self, ctx, vd, additional):
        pass

    def _title(self, ctx):
        return "OA: {:.3f}, RPA: {:.3f}, RCA: {:.3f}, VR: {:.3f}, VFA: {:.3f}, VA: {:.3f}, Loss {:.4f}".format(
            ctx.metrics['Overall Accuracy'],
            ctx.metrics['Raw Pitch Accuracy'],
            ctx.metrics['Raw Chroma Accuracy'],
            ctx.metrics['Voicing Recall'],
            ctx.metrics['Voicing False Alarm'],
            ctx.metrics['Voicing Accuracy'],
            ctx.metrics['Loss']
        )

class EvaluationHook_mf0:
    def _title(self, ctx):
        return "F1: {:.3f}, Acc: {:.3f}, Pr: {:.3f}, Re: {:.3f}, Loss {:.4f}".format(
            ctx.metrics['F1'],
            ctx.metrics['Accuracy'],
            ctx.metrics['Precision'],
            ctx.metrics['Recall'],
            ctx.metrics['Loss'],
        )


class EvaluationHook_mir(EvaluationHook_mf0):
    def _title(self, ctx):
        return "F1: {:.3f}, Pr: {:.3f}, Re: {:.3f}, Loss {:.4f}".format(
            ctx.metrics['micro f1'],
            ctx.metrics['micro precision'],
            ctx.metrics['micro recall'],
            ctx.metrics['Loss'],
        )


class VisualOutputHook(EvaluationHook):
    def __init__(self, draw_notes=True, draw_probs=True, draw_confusion=False, draw_hists=False):
        self.draw_notes = draw_notes
        self.draw_probs = draw_probs
        self.draw_confusion = draw_confusion
        self.draw_hists = draw_hists
        self.markersize = 1

    def before_predict(self, ctx, vd):
        self.ref_notes = []
        self.est_notes = []
        additional = []
        if self.draw_probs:
            additional.append(ctx.note_probabilities)
        return additional

    def every_aa(self, ctx, vd, aa, est):
        est_time, est_notes, est_freq = est
        self.ref_notes += aa.annotation.notes_mf0
        self.est_notes += datasets.common.melody_to_multif0(est_notes)

    def after_run(self, ctx, vd, additional):
        prefix = "valid_{}/".format(vd.name)
        title = self._title(ctx)
        reference = self.ref_notes
        estimation = self.est_notes
        global_step = tf.train.global_step(ctx.session, ctx.global_step)

        if self.draw_notes:
            note_probs = None
            if self.draw_probs:
                note_probs = np.concatenate(list(additional[ctx.note_probabilities].values())).T

            fig = vis.draw_notes(reference, estimation, title=title, note_probs=note_probs, markersize=self.markersize, min_note=ctx.min_note, note_range=ctx.note_range)
            add_fig(fig, ctx.summary_writer, prefix+"notes", global_step)

        if self.draw_confusion:
            fig = vis.draw_confusion(reference, estimation)
            add_fig(fig, ctx.summary_writer, prefix+"confusion", global_step)

        if self.draw_hists:
            fig = vis.draw_hists(reference, estimation)
            add_fig(fig, ctx.summary_writer, prefix+"histograms", global_step)

class CSVOutputWriterHook(EvaluationHook):
    def __init__(self, output_path=None, output_file=None, output_format="mdb"):
        self.output_path = output_path
        self.output_file = output_file
        self.output_format = output_format

    def before_predict(self, ctx, vd):
        self.write_estimations_timer = 0
        return []

    def every_aa(self, ctx, vd, aa, est):
        est_time, est_notes, est_freq = est
        timer = time.time()
        if self.output_file is None:
            est_dir = self.output_path
            if self.output_path is None:
                est_dir = os.path.join(ctx.logdir, ctx.args.checkpoint+"-f0-outputs", vd.name+"-test-melody-outputs")
            os.makedirs(est_dir, exist_ok=True)
            output_file = os.path.join(est_dir, aa.audio.filename+".csv")
        else:
            output_file = self.output_file

        with open(output_file, 'w') as f:
            if self.output_format == "mdb":
                writer = csv.writer(f)
            if self.output_format == "mirex":
                writer = csv.writer(f, delimiter="\t")
            writer.writerows(zip(est_time, est_freq))
        self.write_estimations_timer += time.time()-timer

    def after_run(self, ctx, vd, additional):
        print("csv outputs written in {:.2f}s".format(self.write_estimations_timer))


class CSVBatchOutputWriterHook_mf0(EvaluationHook):
    def __init__(self, split="test", output_reference=False):
        self.split = split
        self.output_reference = output_reference

    def before_predict(self, ctx, vd):
        self.est_dir = os.path.join(ctx.logdir, ctx.args.checkpoint+"-mf0-outputs", vd.name+"-"+self.split+"-outputs")
        os.makedirs(self.est_dir, exist_ok=True)
        if self.output_reference:
            self.ref_dir = os.path.join(ctx.logdir, ctx.args.checkpoint+"-mf0-outputs", vd.name+"-"+self.split+"-outputs", "reference")
            os.makedirs(self.ref_dir, exist_ok=True)

        return []

    def every_aa(self, ctx, vd, aa, est):
        est_time, est_notes, est_freqs = est
        output_file = os.path.join(self.est_dir, aa.audio.filename+".csv")
        with open(output_file, 'w') as f:
            writer = csv.writer(f)
            for t, fs in zip(est_time, est_freqs):
                writer.writerow([t]+list(fs))

        if self.output_reference:
            ref_time = aa.annotation.times
            ref_freqs = aa.annotation.freqs_mf0
            output_file_ref = os.path.join(self.ref_dir, aa.audio.filename+".csv")
            with open(output_file_ref, 'w') as f:
                writer = csv.writer(f)
                for t, fs in zip(ref_time, ref_freqs):
                    writer.writerow([t]+list(fs))


class BatchOutputWriterHook_mir(EvaluationHook):
    def __init__(self, split="test", output_reference=False):
        self.split = split
        self.output_reference = output_reference

    def before_predict(self, ctx, vd):
        self.est_dir = os.path.join(ctx.logdir, ctx.args.checkpoint+"-mir-outputs", vd.name+"-"+self.split+"-outputs")
        os.makedirs(self.est_dir, exist_ok=True)
        if self.output_reference:
            self.ref_dir = os.path.join(ctx.logdir, ctx.args.checkpoint+"-mir-outputs", vd.name+"-"+self.split+"-outputs", "reference")
            os.makedirs(self.ref_dir, exist_ok=True)

        return []

    def every_aa(self, ctx, vd, aa, est):
        est_time, classes, est_matrix = est
        ref_matrix = aa.annotation.ref_matrix(ctx.note_range, 0)

        output_file = os.path.join(self.est_dir, aa.audio.filename+".npy")
        np.save(output_file, est_matrix)

        if self.output_reference:
            output_file_ref = os.path.join(self.ref_dir, aa.audio.filename+".npy")
            np.save(output_file_ref, ref_matrix)


class MetricsHook(EvaluationHook):
    def __init__(self, write_summaries=True, print_detailed=False, split="valid"):
        self.print_detailed = print_detailed
        self.write_summaries = write_summaries
        self.split = split

    def before_predict(self, ctx, vd):
        self.all_metrics = []
        return [ctx.loss]

    def every_aa(self, ctx, vd, aa, est):
        est_time, est_notes, est_freq = est
        ref_time = aa.annotation.times
        ref_freq = aa.annotation.freqs[:, 0]

        assert len(ref_time) == len(est_time)
        assert len(ref_freq) == len(est_freq)
        assert len(ref_freq) == len(ref_time)

        metrics = mir_eval.melody.evaluate(ref_time, ref_freq, est_time, est_freq)

        ref_v = ref_freq > 0
        est_v = est_freq > 0

        cent_voicing = mir_eval.melody.to_cent_voicing(ref_time, ref_freq, est_time, est_freq)
        metrics["Raw Pitch Accuracy 25 cent"] = mir_eval.melody.raw_chroma_accuracy(*cent_voicing, cent_tolerance=25)
        metrics["Raw Chroma Accuracy 25 cent"] = mir_eval.melody.raw_pitch_accuracy(*cent_voicing, cent_tolerance=25)
        metrics["Raw Pitch Accuracy 10 cent"] = mir_eval.melody.raw_chroma_accuracy(*cent_voicing, cent_tolerance=10)
        metrics["Raw Chroma Accuracy 10 cent"] = mir_eval.melody.raw_pitch_accuracy(*cent_voicing, cent_tolerance=10)

        est_freq, est_v = mir_eval.melody.resample_melody_series(est_time, est_freq, est_v, ref_time, "linear")

        metrics["Raw 2 Harmonic Accuracy"] = evaluation.melody.raw_harmonic_accuracy(ref_v, ref_freq, est_v, est_freq, harmonics=2)
        metrics["Raw 3 Harmonic Accuracy"] = evaluation.melody.raw_harmonic_accuracy(ref_v, ref_freq, est_v, est_freq, harmonics=3)
        metrics["Raw 4 Harmonic Accuracy"] = evaluation.melody.raw_harmonic_accuracy(ref_v, ref_freq, est_v, est_freq, harmonics=4)

        timefreq_series = mir_eval.melody.to_cent_voicing(ref_time, ref_freq, ref_time, est_freq)
        metrics["Overall Chroma Accuracy"] = evaluation.melody.overall_chroma_accuracy(*timefreq_series)

        metrics["Voicing Accuracy"] = evaluation.melody.voicing_accuracy(ref_v, est_v)
        metrics["Voiced Frames Proportion"] = est_v.sum() / len(est_v) if len(est_v) > 0 else 0

        self.all_metrics.append(metrics)

    def _save_metrics(self, ctx, vd, additional):
        ctx.metrics = pandas.DataFrame(self.all_metrics).mean()
        ctx.metrics["Loss"] = np.mean(additional[ctx.loss])

        if self.print_detailed:
            print(ctx.metrics)

        if vd.name is not None and self.write_summaries:
            prefix = "{}_{}/".format(self.split, vd.name)

            global_step = tf.train.global_step(ctx.session, ctx.global_step)

            for name, metric in ctx.metrics.items():
                ctx.summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=prefix+simplify_name(name), simple_value=metric)]), global_step)

    def after_run(self, ctx, vd, additional):
        self._save_metrics(ctx, vd, additional)
        print("{}: {}".format(vd.name, self._title(ctx)))

class MetricsHook_mf0(EvaluationHook_mf0, MetricsHook):
    def every_aa(self, ctx, vd, aa, est):
        est_time, est_notes, est_freqs = est
        ref_time = aa.annotation.times
        ref_freqs = aa.annotation.freqs_mf0

        metrics = mir_eval.multipitch.evaluate(ref_time, ref_freqs, est_time, est_freqs)
        metrics["F1"] = 2 * metrics["Precision"] * metrics["Recall"] / (metrics["Precision"] + metrics["Recall"])

        self.all_metrics.append(metrics)

class MetricsHook_mir(EvaluationHook_mir, MetricsHook):
    def __init__(self, instrument_mappings, **kwds):
        self.instrument_mappings = instrument_mappings
        self.target_names = [None]*(max([im["id"] for im in instrument_mappings.values()]) + 1)
        for _, im in instrument_mappings.items():
            i = im["id"]
            self.target_names[i] = im["instrument"]
        
        super().__init__(**kwds)

    def before_predict(self, ctx, vd):
        self.est_matrices = []
        self.ref_matrices = []

        return super().before_predict(ctx, vd)

    def every_aa(self, ctx, vd, aa, est):
        est_time, classes, est_matrix = est
        ref_time = aa.annotation.times
        ref_matrix = aa.annotation.ref_matrix(ctx.note_range, 0)
        self.est_matrices.append(est_matrix)
        self.ref_matrices.append(ref_matrix)
        # print("est", est_matrix, "\nref:", ref_matrix, "\n\n")

        # assert ref_matrix.shape == est_matrix.shape
        # report = sklearn.metrics.classification_report(ref_matrix, est_matrix, target_names=self.target_names, output_dict=True)
        # print(aa.audio.uid)
        # print(sklearn.metrics.classification_report(ref_matrix, est_matrix, target_names=self.target_names))
        # # print(report)
        # metrics = {
        #     "F1": report["micro avg"]["f1-score"],
        #     "Precision": report["micro avg"]["precision"],
        #     "Recall": report["micro avg"]["recall"],
        # }

        # for target in self.target_names:
        #     if report[target]["support"] == 0:
        #         continue
        #     metrics["Class "+target+" F1"] = report[target]["f1-score"]
        #     metrics["Class "+target+" Recall"] = report[target]["recall"]
        #     metrics["Class "+target+" Precision"] = report[target]["precision"]

        # self.all_metrics.append(metrics)
    def after_run(self, ctx, vd, additional):
        ref_matrix = np.concatenate(self.ref_matrices)
        est_matrix = np.concatenate(self.est_matrices)
        print("ref_matrix.shape", ref_matrix.shape)
        print("est_matrix.shape", est_matrix.shape)
        report = sklearn.metrics.classification_report(ref_matrix, est_matrix, target_names=self.target_names, output_dict=True)
        print(sklearn.metrics.classification_report(ref_matrix, est_matrix, target_names=self.target_names))

        metrics = {
            "micro f1": report["micro avg"]["f1-score"],
            "micro precision": report["micro avg"]["precision"],
            "micro recall": report["micro avg"]["recall"],
            "macro f1": report["macro avg"]["f1-score"],
            "macro precision": report["macro avg"]["precision"],
            "macro recall": report["macro avg"]["recall"],
        }

        for target in self.target_names:
            if report[target]["support"] == 0:
                continue
            metrics["class "+target+" f1"] = report[target]["f1-score"]
            metrics["class "+target+" recall"] = report[target]["recall"]
            metrics["class "+target+" precision"] = report[target]["precision"]

        self.all_metrics.append(metrics)
        return super().after_run(ctx, vd, additional)
    
class SaveSaliencesHook(EvaluationHook):
    def before_predict(self, ctx, vd):
        return [ctx.note_probabilities]

    def after_run(self, ctx, vd, additional):
        timer = time.time()
        est_dir = os.path.join(ctx.logdir, ctx.args.checkpoint+"-f0-saliences", vd.name+"-test-melody-outputs")
        os.makedirs(est_dir, exist_ok=True)

        for uid, salience in additional[ctx.note_probabilities].items():
            aa = vd.dataset.get_annotated_audio_by_uid(uid)
            np.save(os.path.join(est_dir, aa.audio.filename+".npy"), salience)

        print("saliences written in {:.2f}s".format(time.time()-timer))


class VisualOutputHook_mf0(EvaluationHook_mf0, VisualOutputHook):
    def every_aa(self, ctx, vd, aa, est):
        est_time, est_notes, est_freq = est
        self.ref_notes += aa.annotation.notes_mf0
        self.est_notes += est_notes


class VisualOutputHook_mir(EvaluationHook_mir, VisualOutputHook):
    def every_aa(self, ctx, vd, aa, est):
        time, classes, est_matrix = est
        self.ref_notes += aa.annotation.notes_mf0
        self.est_notes += classes
        self.markersize = 20

class SaveBestModelHook(EvaluationHook):
    def __init__(self, logdir, watch_metric = "Raw Pitch Accuracy"):
        self.best_value = -1
        self.logdir = logdir
        self.watch_metric = watch_metric

    def after_run(self, ctx, vd, additional):
        self.model_name = "model-best-{}".format(vd.name)
        best_metrics_csv = os.path.join(self.logdir, self.model_name+".csv")
        if self.best_value == -1 and os.path.isfile(best_metrics_csv):
            self.best_value = pandas.read_csv(best_metrics_csv, header=None, index_col=0, squeeze=True)[self.watch_metric]

        value = ctx.metrics[self.watch_metric]
        if value > self.best_value:
            self.best_value = value
            print("Saving best model, best value = {:.2f}".format(value))
            ctx.save(self.model_name, ctx.saver_best)
            ctx.metrics.to_csv(best_metrics_csv)


class AdjustVoicingHook(EvaluationHook):
    def before_predict(self, ctx, vd):
        return [ctx.est_notes_confidence]

    def after_predict(self, ctx, vd, estimations, additional):
        thresholds = np.arange(0.0, 1.0, 0.01)
        results = []
        for threshold in thresholds:
            threshold_results = []
            for uid, est_notes_confidence in additional[ctx.est_notes_confidence].items():
                aa = vd.dataset.get_annotated_audio_by_uid(uid)
                ref_voicing = aa.annotation.freqs[:,0] > 0
                est_voicing = est_notes_confidence > threshold
                voicing_accuracy = evaluation.melody.voicing_accuracy(ref_voicing, est_voicing)
                threshold_results.append(voicing_accuracy)
            results.append(np.mean(threshold_results))

        best_threshold = thresholds[np.argmax(results)]
        print("Voicing threshold: {:.2f}, best voicing accuracy: {:.3f}".format(best_threshold, np.max(results)))
        ctx.voicing_threshold.load(best_threshold, ctx.session)

        for uid, (est_time, est_note, est_freq) in estimations.items():
            est_notes_confidence = additional[ctx.est_notes_confidence][uid]
            est_voicing = est_notes_confidence > best_threshold
            new_est_freq = np.abs(est_freq) * (est_voicing*2-1)
            estimations[uid] = (est_time, est_note, new_est_freq)


@jit(nopython=True)
def _find_thresholds(note_probs, ref_matrix):
    best_thresholds = np.full((note_probs.shape[1],), 0.5)
    best_results = np.zeros((note_probs.shape[1],))

    for threshold in thresholds:
        threshold_results = []
        est_matrix = note_probs > threshold
        # print("===", threshold, "===")
        tp = np.sum(ref_matrix & est_matrix, axis=0)
        fp = np.sum((~ref_matrix) & est_matrix, axis=0)
        fn = np.sum(ref_matrix & (~est_matrix), axis=0)

        p = tp/(tp + fp)
        r = tp/(tp + fn)
        f1 = (2 * p * r) / (p + r)

        for i, best_result in enumerate(best_results):
            res = f1[i]
            if res > best_result:
                best_results[i] = res
                best_thresholds[i] = threshold

    return best_thresholds, best_results

class AdjustVoicingHook_mir(EvaluationHook):
    def before_predict(self, ctx, vd):
        return [ctx.note_probabilities]

    def after_predict(self, ctx, vd, estimations, additional):
        thresholds = np.arange(0.0, 1.0, 0.05)
        note_probs_list = []
        ref_matrix_list = []

        for uid, note_probs in additional[ctx.note_probabilities].items():
            aa = vd.dataset.get_annotated_audio_by_uid(uid)
            ref_matrix = aa.annotation.ref_matrix(ctx.note_range, 0)
            note_probs_list.append(note_probs)
            ref_matrix_list.append(ref_matrix)
        
        note_probs = np.concatenate(note_probs_list)
        ref_matrix = np.concatenate(ref_matrix_list)

        # print(note_probs.shape, ref_matrix.shape)
        best_thresholds = np.full((note_probs.shape[1],), 0.5)
        best_results = np.zeros((note_probs.shape[1],))


        for threshold in thresholds:
            threshold_results = []
            est_matrix = note_probs > threshold
            # print("===", threshold, "===")
            tp = np.sum(ref_matrix & est_matrix, axis=0)
            fp = np.sum((~ref_matrix) & est_matrix, axis=0)
            fn = np.sum(ref_matrix & (~est_matrix), axis=0)

            p = tp/(tp + fp)
            r = tp/(tp + fn)
            f1 = (2 * p * r) / (p + r)

            # print(f1)

            # results = sklearn.metrics.classification_report(ref_matrix, est_matrix, output_dict=False)
            # print(results)
            for i, best_result in enumerate(best_results):
                res = f1[i]
                if res > best_result:
                    best_results[i] = res
                    best_thresholds[i] = threshold
            
            # print(best_thresholds)
            # print(best_results)
            # results.append(np.mean(threshold_results))

        print("Voicing thresholds: ", best_thresholds, ", best f1 scores: ", best_results)
        ctx.thresholds = best_thresholds

        for uid, (est_time, classes, est_matrix) in estimations.items():
            note_probs = additional[ctx.note_probabilities][uid]
            new_est_matrix = note_probs > best_thresholds
            new_classes = _process_fetched_values(note_probs, best_thresholds)
            estimations[uid] = (est_time, new_classes, new_est_matrix)
