import numpy as np
import tensorflow as tf

import datasets
from .network_multif0 import NetworkMultif0, safe_div
from numba import jit

@jit(nopython=True)
def _process_fetched_values(probs, thresholds):
    notes = []
    for i in range(len(probs)):
        frame = probs[i]
        notes_in_frame = np.array([i for i, prob in enumerate(frame) if prob > thresholds[i]])
        notes.append(notes_in_frame)

    return notes

class NetworkMultiInstrumentRecognition(NetworkMultif0):
    def __init__(self, args, seed=42):
        self.thresholds = np.full((args.note_range,), 0.5)
        super().__init__(args, seed=seed)
    
    def construct(self, args, create_model, output_types, output_shapes, create_summaries=None, spectrogram_info=None, class_weights=None):
        self.class_weights = tf.constant(class_weights)
        return super().construct(args, create_model, output_types, output_shapes, create_summaries=create_summaries, spectrogram_info=spectrogram_info)

    def _process_estimations(self, fetched_values):
        times_per_uid = fetched_values[self.times]
        note_probabilities_per_uid = fetched_values[self.note_probabilities]
        estimations = {}
        # !!! TODO: variable note probability threshold
        for uid, probs in note_probabilities_per_uid.items():
            time = times_per_uid[uid]
            classes = _process_fetched_values(probs, self.thresholds)
            estimations[uid] = (time, classes, probs > self.thresholds)
        
        return estimations
