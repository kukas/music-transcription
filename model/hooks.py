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
    def before_run(self, ctx, vd):
        pass

    def every_aa(self, ctx, vd, aa, est_time, est_freq):
        pass

    def after_run(self, ctx, vd, additional):
        pass

    def _title(self, ctx):
        return "OA: {:.3f}, RPA: {:.3f}, RCA: {:.3f}, VR: {:.3f}, VFA: {:.3f}, Loss {:.4f}".format(
            ctx.metrics['Overall Accuracy'],
            ctx.metrics['Raw Pitch Accuracy'],
            ctx.metrics['Raw Chroma Accuracy'],
            ctx.metrics['Voicing Recall'],
            ctx.metrics['Voicing False Alarm'],
            ctx.metrics['Loss']
        )


class EvaluationHook_mf0:
    def _title(self, ctx):
        return "Acc: {:.3f}, Pr: {:.3f}, Re: {:.3f}, Sub: {:.3f}".format(
            ctx.metrics['Accuracy'],
            ctx.metrics['Precision'],
            ctx.metrics['Recall'],
            ctx.metrics['Substitution Error'],
        )


class VisualOutputHook(EvaluationHook):
    def __init__(self, draw_notes=True, draw_probs=True, draw_confusion=False, draw_hists=False):
        self.draw_notes = draw_notes
        self.draw_probs = draw_probs
        self.draw_confusion = draw_confusion
        self.draw_hists = draw_hists

    def before_run(self, ctx, vd):
        self.reference = []
        self.estimation = []
        additional = []
        if self.draw_probs:
            additional.append(ctx.note_probabilities)
        return additional

    def every_aa(self, ctx, vd, aa, est_time, est_freq):
        self.reference += aa.annotation.notes_mf0
        self.estimation.append(datasets.common.hz_to_midi_safe(est_freq))

    def after_run(self, ctx, vd, additional):
        prefix = "valid_{}/".format(vd.name)
        title = self._title(ctx)
        reference = self.reference
        estimation = datasets.common.melody_to_multif0(np.concatenate(self.estimation))
        global_step = tf.train.global_step(ctx.session, ctx.global_step)

        if self.draw_notes:
            note_probs = None
            if self.draw_probs:
                note_probs = np.concatenate(list(additional[ctx.note_probabilities].values())).T

            fig = vis.draw_notes(reference, estimation, title=title, note_probs=note_probs)
            add_fig(fig, ctx.summary_writer, prefix+"notes", global_step)

        if self.draw_confusion:
            fig = vis.draw_confusion(reference, estimation)
            add_fig(fig, ctx.summary_writer, prefix+"confusion", global_step)

        if self.draw_hists:
            fig = vis.draw_hists(reference, estimation)
            add_fig(fig, ctx.summary_writer, prefix+"histograms", global_step)


class MetricsHook(EvaluationHook):
    def __init__(self, write_summaries=True, print_detailed=False, write_estimations=False):
        self.print_detailed = print_detailed
        self.write_summaries = write_summaries
        self.write_estimations = write_estimations

    def before_run(self, ctx, vd):
        self.write_estimations_timer = 0
        self.all_metrics = []
        return [ctx.loss]

    def every_aa(self, ctx, vd, aa, est_time, est_freq):
        if self.write_estimations:
            timer = time.time()
            est_dir = os.path.join(ctx.logdir, ctx.args.checkpoint+"-f0-outputs", vd.name+"-test-melody-outputs")
            os.makedirs(est_dir, exist_ok=True)
            with open(os.path.join(est_dir, aa.audio.filename+".csv"), 'w') as f:
                writer = csv.writer(f)
                writer.writerows(zip(est_time, est_freq))
            self.write_estimations_timer += time.time()-timer

        ref_time = aa.annotation.times
        ref_freq = np.squeeze(aa.annotation.freqs, 1)

        assert len(ref_time) == len(est_time)
        assert len(ref_freq) == len(est_freq)
        assert len(ref_freq) == len(est_freq)

        metrics = mir_eval.melody.evaluate(ref_time, ref_freq, est_time, est_freq)

        ref_v = ref_freq > 0
        est_v = est_freq > 0

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
            prefix = "valid_{}/".format(vd.name)

            global_step = tf.train.global_step(ctx.session, ctx.global_step)

            for name, metric in ctx.metrics.items():
                ctx.summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=prefix+simplify_name(name), simple_value=metric)]), global_step)

    def after_run(self, ctx, vd, additional):
        self._save_metrics(ctx, vd, additional)
        if self.write_estimations:
            print("csv outputs written in {:.2f}s".format(self.write_estimations_timer))
        print("{}: {}".format(vd.name, self._title(ctx)))


class MetricsHook_mf0(EvaluationHook_mf0, MetricsHook):
    def every_aa(self, ctx, vd, aa, est_time, est_freq):
        est_freqs = datasets.common.melody_to_multif0(est_freq)

        ref_time = aa.annotation.times
        ref_freqs = aa.annotation.freqs_mf0

        metrics = mir_eval.multipitch.evaluate(ref_time, ref_freqs, est_time, est_freqs)

        self.all_metrics.append(metrics)


class VisualOutputHook_mf0(EvaluationHook_mf0, VisualOutputHook):
    pass


class SaveBestModelHook(EvaluationHook):
    def __init__(self, logdir):
        self.best_value = -1
        self.logdir = logdir
        self.watch_metric = "Overall Accuracy"

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
