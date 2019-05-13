import librosa
import librosa.display

from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

import tensorflow as tf

import datasets

from IPython.display import Audio
import itertools

def audioplayer(path):
    y, fs = librosa.load(path, sr=None)
    return Audio(y, rate=fs)


def samplesplayer(samples, fs):
    return Audio(samples, rate=fs)


def flatten(notesets, timescale=1):
    indices = [i*timescale for i, notes in enumerate(notesets) for n in notes]
    flatnotes = [n for notes in notesets for n in notes]
    return indices, flatnotes

def draw_notes_on_ax(ref, est, ax, onlyvoiced=False, timescale=1, ms=1):
    def octave_correct(n_ref, n_est):
        note_diff = abs(n_est - n_ref)
        nearest_octave = 12 * np.round(note_diff/12)
        return abs(note_diff - nearest_octave) < 0.5

    def abs_correct(n_ref, n_est):
        return abs(n_est - n_ref) < 0.5

    indices_correct_negative, correct_negative = flatten([[-n_est for n_est in fest if n_est < 0 and any([abs_correct(n_ref, -n_est) for n_ref in fref])] for fref, fest in zip(ref, est)], timescale)
    indices_unvoiced_correct_negative, unvoiced_correct_negative = flatten([[-n_est for n_est in fest if n_est < 0 and len(fref) == 0] for fref, fest in zip(ref, est)], timescale)
    indices_incorrect_negative, incorrect_negative = flatten([[-n_est for n_est in fest if n_est < 0 and (all([not abs_correct(n_ref, -n_est)
                                                                                                               for n_ref in fref]) or len(fref) == 0) and len(fref) > 0] for fref, fest in zip(ref, est)], timescale)
    indices_incorrect_chroma_negative, incorrect_chroma_negative = flatten([[-n_est for n_est in fest if n_est < 0 and all(
        [not abs_correct(n_ref, -n_est) and octave_correct(n_ref, -n_est) for n_ref in fref]) and len(fref) > 0] for fref, fest in zip(ref, est)], timescale)

    indices_correct, correct = flatten([[n_est for n_est in fest if n_est > 0 and any([abs_correct(n_ref, n_est) for n_ref in fref])] for fref, fest in zip(ref, est)], timescale)
    indices_unvoiced_incorrect, unvoiced_incorrect = flatten([[n_est for n_est in fest if n_est > 0 and len(fref) == 0] for fref, fest in zip(ref, est)], timescale)
    indices_incorrect_chroma, incorrect_chroma = flatten([[n_est for n_est in fest if n_est > 0 and all(
        [not abs_correct(n_ref, n_est) and octave_correct(n_ref, n_est) for n_ref in fref]) and len(fref) > 0] for fref, fest in zip(ref, est)], timescale)
    indices_incorrect, incorrect = flatten([[n_est for n_est in fest if n_est > 0 and all([not abs_correct(n_ref, n_est) and not octave_correct(n_ref, n_est)
                                                                                           for n_ref in fref]) and len(fref) > 0] for fref, fest in zip(ref, est)], timescale)
    indices_ref_rest, ref_rest = flatten(ref, timescale)

    style = "."
    ax.plot(indices_ref_rest, ref_rest, style, color="#161925", label="Reference", markersize=ms)
    ax.plot(indices_correct, correct, style, color="#02C39A", label="Výška správná; Detekce správná pozitivní", markersize=ms)
    if not onlyvoiced:
        ax.plot(indices_unvoiced_incorrect, unvoiced_incorrect, style, color="#a0247b", label="Výška nedefinovaná; Detekce nesprávná pozitivní", markersize=ms)
    ax.plot(indices_incorrect_chroma, incorrect_chroma, style, color="#F4A745", label="Výška nesprávná (o oktávu); Detekce správná pozitivní", markersize=ms)
    ax.plot(indices_incorrect, incorrect, style, color="#FF4365", label="Výška nesprávná; Detekce správná pozitivní", markersize=ms)

    if indices_correct_negative:
        ax.plot(indices_correct_negative, correct_negative, style, color="#D1F4EC", label="Výška správná; Detekce nesprávná negativní", markersize=1)
    if indices_unvoiced_correct_negative and not onlyvoiced:
        ax.plot(indices_unvoiced_correct_negative, unvoiced_correct_negative, style, color="#247BA0", label="Výška nedefinovaná; Detekce správná negativní", markersize=1)
    if indices_incorrect_negative:
        ax.plot(indices_incorrect_negative, incorrect_negative, style, color="#FFCBD5", label="Výška nesprávná; Detekce nesprávná negativní", markersize=1)
    if indices_incorrect_chroma_negative:
        ax.plot(indices_incorrect_chroma_negative, incorrect_chroma_negative, style, color="#FADEB9", label="Výška nesprávná (o oktávu); Detekce nesprávná negativní", markersize=1)


def draw_notes(ref, est, title=None, dynamic_figsize=True, note_probs=None):
    nrows = 1
    if note_probs is not None:
        nrows = 2

    width = 9
    height = 6
    if dynamic_figsize:
        width = len(ref)/150
        height = 10

    fig, axs = plt.subplots(nrows, 1, sharex=True, sharey=False, squeeze=False, figsize=(width, height))
    axs = axs[:,0]

    draw_notes_on_ax(ref, est, ax=axs[0])
    axs[0].legend(markerscale=10)

    if title:
        axs[0].set_title(title)
    axs[0].set_ylabel("midi note")
    axs[0].set_xlim(0, len(ref))

    bottom, top = axs[0].get_ylim()
    axs[0].set_ylim(max(0, bottom), min(128, top))

    if note_probs is not None:
        axs[1].set_ylabel("midi note")
        axs[1].set_xlabel("frame")
        axs[1].set_ylim(0, 128)
        axs[1].imshow(note_probs, aspect="auto", origin='lower', extent=[0, note_probs.shape[1], 0, 128])

    # indices_ref, ref = flatten(ref)
    # axs[1].plot(indices_ref, ref, ",", color="#ff0000", alpha=1.0)

    plt.tight_layout()

    return fig


def draw_confusion(ref, est):
    fig, ax = plt.subplots(figsize=(15, 15))
    cm = np.zeros((128, 128))

    for fref, fest in zip(ref, est):
        n_ref = int(np.round(fref[0])) if len(fref) > 0 else 0
        n_est = np.abs(int(np.round(fest[0]))) if len(fest) > 0 else 0
        if n_ref == 0 or n_est == 0:
            continue
        cm[n_est, n_ref] += 1

    # cm /= len(ref)

    xticks = np.arange(128)
    yticks = np.arange(128)

    indices_0 = np.nonzero(np.sum(cm, 0))[0]
    indices_1 = np.nonzero(np.sum(cm, 1))[0]

    if len(indices_0) >= 2:
        first_nonzero = indices_0[0]
        last_nonzero = indices_0[-1] + 1
        cm = cm[:,first_nonzero:last_nonzero]
        xticks = xticks[first_nonzero:last_nonzero]

    if len(indices_1) >= 2:
        first_nonzero = indices_1[0]
        last_nonzero = indices_1[-1] + 1
        cm = cm[first_nonzero:last_nonzero, :]
        yticks = yticks[first_nonzero:last_nonzero]

    ax.imshow(cm)
    # ax.title(title)
    # ax.colorbar()
    
    plt.xticks(np.arange(len(xticks)), xticks, rotation=45)
    plt.yticks(np.arange(len(yticks)), yticks)

    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     if cm[i, j] != 0:
    #         ax.text(j, i, format(cm[i, j], ".2f")[1:],
    #                 horizontalalignment="center",
    #                 color="black" if cm[i, j] > thresh else "white")

    # plt.grid(which="minor", color="w", linestyle='-', linewidth=3)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return fig

def draw_hists(ref, est):
    ref_note = datasets.common.multif0_to_melody(ref)
    est_note = np.abs(datasets.common.multif0_to_melody(est))
    diff = est_note - ref_note

    fig, axs = plt.subplots(2, 1, figsize=(15, 6))
    axs[0].grid(True, axis="y", zorder=0)
    axs[1].grid(True, zorder=0)
    axs[0].set_title("histogram of estimation distances from ground truth")
    axs[0].set_xlabel("distance in semitones")
    axs[0].set_ylabel("number of notes")

    # set for MedleyDB, TODO: more general setting
    axs[0].set_ylim(0, 15000)

    bins = np.arange(-24, 25)
    axs[0].set_xticks(bins)
    bins = bins-0.5
    # ignore the correct estimation bin
    diff = diff[np.abs(diff) > 0.5]
    axs[0].hist(diff, bins=bins, zorder=3)

    correct_notes = np.zeros([128])
    total_notes = np.zeros([128]) + 0.00001  # divide by zero fix
    for n_ref, n_est in zip(ref_note, est_note):
        if n_ref == 0 or n_est == 0:
            continue
        if abs(n_ref-n_est) < 0.5:
            correct_notes[int(round(n_ref))] += 1
        total_notes[int(round(n_ref))] += 1

    axs[1].set_title("pitch accuracy for every note class")
    axs[1].set_xlabel("midi note")
    axs[1].set_ylabel("accuracy")
    axs[1].set_ylim(0, 1)
    axs[1].bar(np.arange(len(correct_notes)), correct_notes/total_notes, zorder=3)

    plt.tight_layout()

    return fig

def draw_probs(probs, ref, est, title=None):
    fig, ax = plt.subplots(figsize=(len(ref)/100, 6))
    # ax.set_ylim(0, 128)
    if title:
        ax.set_title(title)
    ax.set(xlabel='frame', ylabel='midi note')

    ax.imshow(probs, aspect="auto", origin='lower', extent=[0, probs.shape[1], 0, 128])

    indices_ref, ref = flatten(ref)
    ax.plot(indices_ref, ref, ",", color="#ff0000", alpha=1.0)

    plt.tight_layout()

    return fig


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    data = np.roll(data, 3, axis=2)
    return data


def fig2summary(fig):
    # Write the image to a string
    s = BytesIO()
    fig.savefig(s, format='png')
    shape = fig.canvas.get_width_height()
    img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                               width=shape[0], height=shape[1])

    return img_sum


def draw_spectrum(audio, samplerate):
    fig, ax = plt.subplots()

    f, t, X = scipy.signal.stft(audio, samplerate, nperseg=2048, noverlap=2048-256)
    S = librosa.amplitude_to_db(np.abs(X))

    ax.set_yscale('log')
    ax.pcolormesh(t, f, S, cmap="inferno")
    ax.set_ylim(27.5, 4400)

    return fig


def draw_cqt(audio, samplerate):
    s = 3
    C = librosa.cqt(audio, sr=samplerate, n_bins=60 * s, bins_per_octave=12 * s, hop_length=16)
    librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                             sr=samplerate, x_axis='time', y_axis='cqt_note')
