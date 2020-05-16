import pytest
from . import melody
import numpy as np
import mir_eval

def test_raw_harmonic_accuracy():
    ref_freq = np.array([0, 440, 440, 440, 440])
    ref_voicing = ref_freq > 0

    est_freq = np.array([0, 0, 0, 0, 0])
    est_voicing = est_freq > 0

    score = melody.raw_harmonic_accuracy(ref_voicing, ref_freq, est_voicing, est_freq)
    assert np.allclose(0.0, score)

    est_freq = np.array([0, 430, 660, 890, 1760])
    est_voicing = est_freq > 0

    score = melody.raw_harmonic_accuracy(ref_voicing, ref_freq, est_voicing, est_freq)
    assert np.allclose(0.75, score)

    score = melody.raw_harmonic_accuracy(ref_voicing, ref_freq, est_voicing, est_freq, harmonics=3)
    assert np.allclose(0.5, score)

def test_overall_chroma_accuracy():
    ref_cent = np.array([0, 0, 1195, 1800, 2405])
    est_cent = np.array([0, 0, 1200, 1200, 1200])
    ref_voicing = ref_cent>0
    est_voicing = est_cent>0
    score = melody.overall_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
    assert np.allclose(0.8, score)

    # ref_cent = np.array([0, 0, 1395, 1800, 2605])
    ref_cent = np.array([0, 0, 1195, 1800, 2605])
    est_cent = np.array([0, 0, 0, 0, 0])
    ref_voicing = ref_cent > 0
    est_voicing = est_cent > 0
    score = melody.overall_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
    # error in older version of mir_eval
    if mir_eval.__version__ > "0.5":
        assert np.allclose(0.4, score)

def test_voicing_accuracy():
    ref = np.array([])
    est = np.array([])
    score = melody.voicing_accuracy(ref, est)
    assert np.allclose(0.0, score)

    ref = np.array([0,0])
    est = np.array([1])
    with pytest.raises(ValueError):
        melody.voicing_accuracy(ref, est)

    ref = np.array([1, 1, 0, 0], dtype=np.bool)
    est = np.array([0, 1, 0, 1], dtype=np.bool)
    score = melody.voicing_accuracy(ref, est)

    assert np.allclose(0.5, score)

    ref = np.array([1, 1, 0, 0], dtype=np.bool)
    est = np.array([1, 1, 0, 0], dtype=np.bool)
    score = melody.voicing_accuracy(ref, est)

    assert np.allclose(1.0, score)

    ref = np.array([0, 0, 1, 1], dtype=np.bool)
    est = np.array([1, 1, 0, 0], dtype=np.bool)
    score = melody.voicing_accuracy(ref, est)

    assert np.allclose(0.0, score)
