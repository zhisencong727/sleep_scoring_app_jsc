# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 2024

@author: yzhao and zcong
"""

import math
from fractions import Fraction

import numpy as np
from scipy import stats
from scipy import signal
from scipy.io import loadmat


def trim_missing_labels(filt, trim="b"):
    first = 0
    trim = trim.upper()
    if "F" in trim:
        for i in filt:
            if i == -1 or np.isnan(i):
                first = first + 1
            else:
                break
    last = len(filt)
    if "B" in trim:
        for i in filt[::-1]:
            if i == -1 or np.isnan(i):
                last = last - 1
            else:
                break
    return filt[first:last]


def reshape_sleep_data_ne(mat, segment_size=512, standardize=False, has_labels=True):
    eeg = mat["eeg"].flatten()
    emg = mat["emg"].flatten()
    ne = mat["ne"].flatten()

    if standardize:
        eeg = stats.zscore(eeg)
        emg = stats.zscore(emg)
        ne = stats.zscore(ne)

    eeg_freq = mat["eeg_frequency"].item()
    ne_freq = mat["ne_frequency"].item()

    # clip the last non-full second and take the shorter duration of the two
    original_end_time = min(math.floor(eeg.size / eeg_freq),math.floor(ne.size/ne_freq))


    # if sampling rate is much higher than 512, downsample using poly resample
    if math.ceil(eeg_freq) != segment_size and math.floor(eeg_freq) != segment_size:
        down, up = (
            Fraction(eeg_freq / segment_size).limit_denominator(100).as_integer_ratio()
        )
        print(f"file has sampling frequency of {eeg_freq}.")
        eeg = signal.resample_poly(eeg, up, down)
        emg = signal.resample_poly(emg, up, down)
        eeg_freq = segment_size
    

    # recalculate end time after upsampling ne
    resampled_end_time_eeg = math.floor(len(eeg) / eeg_freq)
    resampled_end_time_ne = math.floor(len(ne) / ne_freq)
    end_time = min(resampled_end_time_eeg, resampled_end_time_ne)

    time_sec = np.arange(end_time)
    start_indices = np.ceil(time_sec * eeg_freq).astype(int)

    # Reshape start_indices to be a column vector (N, 1)
    start_indices = start_indices[:, np.newaxis]
    segment_array = np.arange(segment_size)
    # Use broadcasting to add the range_array to each start index
    indices = start_indices + segment_array
    
    segment_size_ne = 10
    segment_array_ne = np.arange(segment_size_ne)
    ne_start_indices = np.ceil(time_sec * ne_freq).astype(int)
    ne_start_indices = ne_start_indices[:, np.newaxis]
    max_ne_start_index = len(ne) - segment_size
    #ne_start_indices = ne_start_indices[ne_start_indices[:, 0] <= max_ne_start_index]
    ne_indices = ne_start_indices + segment_array_ne
    
    eeg_reshaped = eeg[indices]
    emg_reshaped = emg[indices]
    ne_reshaped = ne[ne_indices]
    #print("EEG_reshape.shape:",eeg_reshaped.shape)
    #print("EMG_reshape.shape:",emg_reshaped.shape)
    #print("NE_reshape.shape:",ne_reshaped.shape)
    return eeg_reshaped, emg_reshaped, ne_reshaped


if __name__ == "__main__":
    path = "C:/Users/yzhao/python_projects/sleep_scoring/user_test_files/"
    mat_file = path + "sal_588.mat"
    mat = loadmat(mat_file)
    eeg_reshaped, emg_reshaped, ne_reshaped, sleep_scores = reshape_sleep_data_ne(mat)