import librosa
import librosa.display

from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

import tensorflow as tf

from IPython.display import Audio

def audioplayer(path):
    y, fs = librosa.load(path, sr=None)
    return Audio(y, rate=fs)

def samplesplayer(samples, fs):
    return Audio(samples, rate=fs)

def flatten(notesets):
    indices = [i for i, notes in enumerate(notesets) for n in notes]
    flatnotes = [n for notes in notesets for n in notes]
    return indices, flatnotes

def draw_notes(ref, est, style = ".", title = None):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_ylim(0,128)
    if title:
        ax.set_title(title)
    ax.set(xlabel='frame', ylabel='midi note')

    # ref = np.array(ref, dtype=np.float16)
    # est = np.array(est, dtype=np.float16)
    
    indices_correct, correct = flatten([[n_est for n_est in fest if any([abs(n_est - n_ref) < 0.5 for n_ref in fref])] for fref, fest in zip(ref, est)])
    indices_incorrect, incorrect = flatten([[n_est for n_est in fest if all([abs(n_est - n_ref) >= 0.5 for n_ref in fref])] for fref, fest in zip(ref, est)])
    indices_ref_rest, ref_rest = flatten([[n_ref for n_ref in fref if all([abs(n_est - n_ref) >= 0.5 for n_est in fest])] for fref, fest in zip(ref, est)])

    ax.plot(indices_ref_rest, ref_rest, style, color="#222222", label="REF")
    ax.plot(indices_incorrect, incorrect, style, color="#ff300e", label="EST incorrect")
    ax.plot(indices_correct, correct, style, color="#0ab02d", label="EST correct")

    legend = ax.legend()

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
    ax.set_ylim(27.5,4400)

def draw_cqt(audio, samplerate):
    s = 3
    C = librosa.cqt(audio, sr=samplerate, n_bins=60 * s, bins_per_octave=12 * s, hop_length=16)
    librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                             sr=samplerate, x_axis='time', y_axis='cqt_note')
