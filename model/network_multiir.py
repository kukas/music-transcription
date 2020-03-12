import numpy as np
import tensorflow as tf

import datasets
from .network_multif0 import NetworkMultif0, safe_div
from numba import jit

@jit(nopython=True)
def _process_fetched_values(probs):
    notes = []
    for i in range(len(probs)):
        frame = probs[i]
        notes_in_frame = np.array([i for i, prob in enumerate(frame) if prob > 0.5])
        notes.append(notes_in_frame)

    return notes

class NetworkMultiInstrumentRecognition(NetworkMultif0):
    def _process_estimations(self, fetched_values):
        times_per_uid = fetched_values[self.times]
        note_probabilities_per_uid = fetched_values[self.note_probabilities]
        estimations = {}
        # !!! TODO: variable note probability threshold
        for uid, probs in note_probabilities_per_uid.items():
            time = times_per_uid[uid]
            classes = _process_fetched_values(probs)
            estimations[uid] = (time, classes, probs > 0.5)
        
        return estimations
