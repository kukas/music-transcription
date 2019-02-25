import librosa
import librosa.display

from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

import tensorflow as tf

from IPython.display import Audio
import itertools

def audioplayer(path):
    y, fs = librosa.load(path, sr=None)
    return Audio(y, rate=fs)


def samplesplayer(samples, fs):
    return Audio(samples, rate=fs)


def flatten(notesets):
    indices = [i for i, notes in enumerate(notesets) for n in notes]
    flatnotes = [n for notes in notesets for n in notes]
    return indices, flatnotes


def draw_notes(ref, est, style=".", title=None):
    fig, ax = plt.subplots(figsize=(15, 6))
    if title:
        ax.set_title(title)
    ax.set(xlabel='frame', ylabel='midi note')

    # ref = np.array(ref, dtype=np.float16)
    # est = np.array(est, dtype=np.float16)

    def octave_correct(n_ref, n_est):
        note_diff = abs(n_est - n_ref)
        nearest_octave = 12 * np.round(note_diff/12)
        return abs(note_diff - nearest_octave) < 0.5

    def abs_correct(n_ref, n_est):
        return abs(n_est - n_ref) < 0.5

    indices_correct_negative, correct_negative = flatten([[-n_est for n_est in fest if n_est < 0 and any([abs_correct(n_ref, -n_est) for n_ref in fref])] for fref, fest in zip(ref, est)])
    indices_incorrect_negative, incorrect_negative = flatten([[-n_est for n_est in fest if n_est < 0 and (all([not abs_correct(n_ref, -n_est)
                                                                                                               for n_ref in fref]) or len(fref) == 0)] for fref, fest in zip(ref, est)])

    indices_correct, correct = flatten([[n_est for n_est in fest if n_est > 0 and any([abs_correct(n_ref, n_est) for n_ref in fref])] for fref, fest in zip(ref, est)])
    indices_unvoiced_incorrect, unvoiced_incorrect = flatten([[n_est for n_est in fest if n_est > 0 and len(fref) == 0] for fref, fest in zip(ref, est)])
    indices_incorrect_chroma, incorrect_chroma = flatten([[n_est for n_est in fest if n_est > 0 and all(
        [not abs_correct(n_est, n_ref) and octave_correct(n_ref, n_est) for n_ref in fref]) and len(fref) > 0] for fref, fest in zip(ref, est)])
    indices_incorrect, incorrect = flatten([[n_est for n_est in fest if n_est > 0 and all([not abs_correct(n_ref, n_est) and not octave_correct(n_ref, n_est)
                                                                                           for n_ref in fref]) and len(fref) > 0] for fref, fest in zip(ref, est)])
    indices_ref_rest, ref_rest = flatten(ref)

    ms = 2
    ax.plot(indices_ref_rest, ref_rest, style, color="#222222", label="REF", markersize=ms)
    ax.plot(indices_unvoiced_incorrect, unvoiced_incorrect, style, color="#A3473A", label="EST voicing error", markersize=ms)
    ax.plot(indices_incorrect_chroma, incorrect_chroma, style, color="#ffb030", label="EST octave error", markersize=ms)
    ax.plot(indices_incorrect, incorrect, style, color="#ff300e", label="EST incorrect", markersize=ms)
    ax.plot(indices_correct, correct, style, color="#0ab02d", label="EST correct", markersize=ms)

    if indices_correct_negative:
        ax.plot(indices_correct_negative, correct_negative, style, color="#0000ff", label="EST correct negative", markersize=ms)
    if indices_incorrect_negative:
        ax.plot(indices_incorrect_negative, incorrect_negative, style, color="#ff00ff", label="EST incorrect negative", markersize=ms)

    ax.legend()

    bottom, top = ax.get_ylim()
    ax.set_ylim(max(0, bottom), min(128, top))

    plt.tight_layout()

    plt.cla()
    plt.clf()
    plt.close('all')

    return fig


def draw_confusion(ref, est):
    fig, ax = plt.subplots(figsize=(15, 15))
    cm = np.zeros((128, 128))

    for fref, fest in zip(ref, est):
        n_ref = int(np.round(fref[0])) if len(fref) > 0 else 0
        n_est = int(np.round(fest[0])) if len(fest) > 0 else 0
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

    plt.cla()
    plt.clf()
    plt.close('all')

    return fig


def draw_probs(probs, ref, title=None):
    fig, ax = plt.subplots(figsize=(15, 6))
    # ax.set_ylim(0, 128)
    if title:
        ax.set_title(title)
    ax.set(xlabel='frame', ylabel='midi note')

    ax.imshow(probs, aspect="auto", origin='lower', extent=[0, probs.shape[1], 0, 128])

    indices_ref, ref = flatten(ref)
    ax.plot(indices_ref, ref, ",", color="#ff0000", alpha=0.5)

    plt.tight_layout()

    plt.cla()
    plt.clf()
    plt.close('all')

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
