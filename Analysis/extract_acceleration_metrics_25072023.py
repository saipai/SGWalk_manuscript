"""
In this api, the raw data file from witmotion is provided as input

The output is a tuple with following in order:
1) Counts
2) Max total acceleration
3) Absolute max range of motion
"""
# %% imports
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from scipy.fft import rfft, rfftfreq
# from scipy.signal import welch
from scipy.signal import find_peaks

from read_data import f_get_data_for_analysis
# from sklearn.cluster import KMeans
# from extract_angl_velocities import f_get_angl_vel_components
# from extract_angles_16052023 import f_extract_arm_swing
# from extract_angl_vel_metrics_23052023 import f_calculate_angl_vel_metrics
# from extract_angle_metrics_23052023 import f_calculate_angle_metrics


# filter the data
def f_low_pass_filter(signal_to_filt, sampling_rate, cut_off_frequency, filter_order=2):
    from scipy.signal import butter, filtfilt

    norm_cut_off = cut_off_frequency/sampling_rate
    b, a = butter(filter_order, norm_cut_off, 'low')
    filtered_signal = filtfilt(b, a, signal_to_filt)
    return filtered_signal

def f_get_total_acceleration(raw_data, fs, filter = 'YES'):
    
    total_acceleration = []

    if filter.lower() == 'yes':
        cut_off_frequency = 15
        filter_order = 2
        sig_1 = f_low_pass_filter(raw_data['X'], fs, cut_off_frequency, filter_order)
        sig_2 = f_low_pass_filter(raw_data['Y'], fs, cut_off_frequency, filter_order)
        sig_3 = f_low_pass_filter(raw_data['Z'], fs, cut_off_frequency, filter_order)
        total_acceleration = ((np.array(sig_1)**2) + (np.array(sig_2)**2) + (np.array(sig_3)**2))**0.5
    else:
        total_acceleration = ((np.array(raw_data['X'])**2) + (np.array(raw_data['Y'])**2) + (np.array(raw_data['Z'])**2))**0.5

    return total_acceleration.tolist()

def f_extract_acceleration_metrics(data, fs, filter, min_accel_prominence):
    result = {}
    metrics = {}

    result['time'] = data['unix_time']
    result['total_acceleration'] = f_get_total_acceleration(data, fs, filter) 
    result['total_acceleration'] = (np.array(result['total_acceleration']) - 1).tolist()
    peaks_idx, peaks = find_peaks(result['total_acceleration'], min_accel_prominence)
    metrics['total_acceleration'] = {}
    metrics['total_acceleration']['peaks'] = {}
    
    if len(peaks_idx) > 0 :
        metrics['total_acceleration']['peaks']['mean'] =  np.nanmean(np.array(result['total_acceleration'])[peaks_idx])
        metrics['total_acceleration']['peaks']['variance'] =  np.nanvar(np.array(result['total_acceleration'])[peaks_idx])
        metrics['total_acceleration']['peaks']['max'] =  np.nanmax(np.array(result['total_acceleration'])[peaks_idx])
    else:
        metrics['total_acceleration']['peaks']['mean'] =  np.nan
        metrics['total_acceleration']['peaks']['variance'] =  np.nan
        metrics['total_acceleration']['peaks']['max'] =  np.nan            
    counts = [len(peaks_idx)]
    # print(counts)
    metrics['total_acceleration']['time_series'] = result['total_acceleration']
    metrics['total_acceleration']['peaks']['idxs'] = peaks_idx
    metrics['total_acceleration']['peaks']['values'] = np.array(result['total_acceleration'])[peaks_idx].tolist()

    return metrics


if __name__== '__main__':
    # Opening raw data file
    directory = '/1TB/SGWalk_IntraCREATE/Pilot_Data/CSV Data (Rename)/'
    week = 'week1'
    game = 'Arctic Punch'
    participant = 'participant01'
    sensor_label = 'right hand'
    format = '.csv'
    fs = 50
    file_name = directory + week + '_' + participant + '_' + game + format
    f = open(file_name)
    df = pd.read_csv(file_name)
    f.close()
    # sensor_labels = ['left hand', 'left leg', 'right hand', 'right leg', 'waist']
    sensor_labels = ['left hand']
   

    metrics = {}
    for sensor_label in sensor_labels:
        accel_data = f_get_data_for_analysis(df, sensor_label, 'acceleration')
        metrics[sensor_label] = {}
        metrics[sensor_label]  = f_extract_acceleration_metrics(accel_data, fs=50, filter='YES', min_accel_prominence=0.2)

    plt.figure()
    plt.plot(metrics[sensor_label]['total_acceleration']['time_series'])
    plt.scatter(metrics[sensor_label]['total_acceleration']['peaks']['idxs'], metrics[sensor_label]['total_acceleration']['peaks']['values'])
    plt.show()