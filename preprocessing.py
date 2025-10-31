"""
Query-by-Humming Preprocessing
Based on Stasiak (2014) - Section 3

This module implements the 5-step preprocessing pipeline for sung/hummed queries.
"""

import numpy as np
from scipy.signal import medfilt
from typing import Tuple

def preprocess_query(
    pitch_vector: np.ndarray,
    sample_rate: int = 8000,
    frame_size: int = 256,
    T1: float = 24.0, 
    T2: float = 14.0,
    median_filter_order: int = 9
) -> np.ndarray:

    q = pitch_vector.copy()
    q = remove_leading_trailing_zeros(q)
    q = remove_outliers(q, T1)
    q = limit_jumps(q, T2)
    q = fill_unvoiced(q)
    q = medfilt(q, kernel_size=median_filter_order)
    return q


def remove_leading_trailing_zeros(pitch_vector: np.ndarray) -> np.ndarray:
    nonzero_indices = np.nonzero(pitch_vector)[0]

    if len(nonzero_indices) == 0:
        return pitch_vector

    first_idx = nonzero_indices[0]
    last_idx = nonzero_indices[-1]

    return pitch_vector[first_idx:last_idx +1]


def remove_outliers(pitch_vector: np.ndarray, T1: float) -> np.ndarray:
    q = pitch_vector.copy()
    voiced_frames = q[q > 0]
    if len(voiced_frames) == 0:
        return q

    median_pitch = np.median(voiced_frames)

    # for MIDI note numbers, but if working in Hz we'd have semitone_distance = 12 * log2(f1/f2)
    for i in range(len(q)):
        if q[i] > 0:
            distance = abs(q[i] - median_pitch)
            if distance > T1:
                q[i] = 0

    return q 


def limit_jumps(pitch_vector:np.ndarray, T2: float) -> np.ndarray: 
    q = pitch_vector.copy()

    for i in range(1, len(q)):
        if q[i] > 0 and q[i-1] > 0:
            jump = abs(q[i] - q[i-1])
            if jump > T2:
                q[i] = 0

    return q 


def fill_unvoiced(pitch_vector: np.ndarray) -> np.ndarray:
    q = pitch_vector.copy()

    last_voiced_value = 0
    for i in range(len(q)):
        if q[i] > 0:
            last_voiced_value = q[i]
        elif last_voiced_value > 0:
            q[i] = last_voiced_value

    return q  


def compute_initial_transposition(
    query: np.ndarray,
    template: np.ndarray
) -> float:
    
    if len(query) < 3 or len(template) < 3:
        q_mean = np.mean(query[:min(3, len(query))])
        t_mean = np.mean(template[:min(3, len(template))])
    else:
        q_mean = (query[1] + query[2]) / 2.0
        t_mean = (template[1] + template[2]) / 2.0

    d_beg = q_mean - t_mean 

    return d_beg


def apply_transposition(query: np.ndarray, d_beg: float) -> np.ndarray:
    return query - d_beg




