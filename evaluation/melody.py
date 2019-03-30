import mir_eval
import numpy as np

def voicing_accuracy(ref_voicing, est_voicing):
    mir_eval.melody.validate_voicing(ref_voicing, est_voicing)
    ref_voicing = ref_voicing.astype(bool)
    est_voicing = est_voicing.astype(bool)
    # When input arrays are empty, return 0 by special case
    if ref_voicing.size == 0 or est_voicing.size == 0:
        return 0.

    # Count True Positives
    TP = (ref_voicing*est_voicing).sum()
    TN = ((ref_voicing == 0)*(est_voicing == 0)).sum()

    return (TP + TN) / float(ref_voicing.shape[0])


def overall_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance=50):
    ref_voicing = ref_voicing.astype(bool)
    est_voicing = est_voicing.astype(bool)
    if ref_voicing.size == 0 or est_voicing.size == 0:
        return 0.

    raw_chroma = mir_eval.melody.raw_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
    n_voiced = ref_voicing.sum()
    TP = raw_chroma * float(n_voiced)
    TN = ((ref_voicing == 0)*(est_voicing == 0)).sum()
    return (TP + TN) / float(ref_cent.shape[0])


def raw_harmonic_accuracy(ref_voicing, ref_freq, est_voicing, est_freq, harmonics=5, cent_tolerance=50):
    est_freq = np.abs(est_freq)

    mir_eval.melody.validate_voicing(ref_voicing, est_voicing)
    mir_eval.melody.validate(ref_voicing, ref_freq, est_voicing, est_freq)
    ref_voicing = ref_voicing.astype(bool)
    est_voicing = est_voicing.astype(bool)
    # When input arrays are empty, return 0 by special case
    if ref_voicing.size == 0 or est_voicing.size == 0 \
       or ref_freq.size == 0 or est_freq.size == 0:
        return 0.

    # If there are no voiced frames in reference, metric is 0
    if ref_voicing.sum() == 0:
        return 0.

    # Raw harmonic = same as raw pitch except that harmonic errors are ignored.
    closest_multiple = np.floor(est_freq/ref_freq + 0.5)
    harmonic_freq = ref_freq*np.fmin(harmonics, closest_multiple)
    harmonic_cent = mir_eval.melody.hz2cents(harmonic_freq)
    correct_voicing = ref_voicing * est_voicing
    frame_correct = (np.abs(mir_eval.melody.hz2cents(est_freq) - harmonic_cent)[correct_voicing] < cent_tolerance)
    n_voiced = float(ref_voicing.sum())
    raw_harmonic = (frame_correct).sum()/n_voiced

    return raw_harmonic
