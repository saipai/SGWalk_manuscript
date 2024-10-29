# imports
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.fft import rfft, rfftfreq
from scipy.signal import welch
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from scipy.integrate import cumulative_trapezoid
from sklearn.decomposition import PCA


# from extract_angles_16052023 import f_extract_arm_swing
# from extract_angle_metrics_23052023 import f_get_counts
# from extract_angl_velocities import f_get_angl_vel_components
from ZurichMOVE_read_data_gyro import f_extract_raw_gyro_data
from read_data import f_get_data_for_analysis

# fft function
def f_fft(time_series, sampling_rate):
    # Note the extra 'r' at the front
    if np.min(np.shape(time_series))>1 and np.max(np.shape(time_series) != np.min(np.shape(time_series))):
        time_series = time_series[:,0]
    
    time_series = time_series - np.mean(time_series)
    yf = np.abs(rfft(time_series))
    xf = rfftfreq(len(time_series), d=1./ sampling_rate)
    return xf, yf

# low-pass filter the data
def f_low_pass_filter(signal_to_filt, sampling_rate, cut_off_frequency, filter_order=2):
    norm_cut_off = cut_off_frequency/sampling_rate
    b, a = butter(filter_order, norm_cut_off, 'low')
    filtered_signal = filtfilt(b, a, signal_to_filt)
    return filtered_signal

# high-pass filter
def f_high_pass_filter(signal_to_filt, sampling_rate, high_pass_cut_off, high_pass_filter_order):
    norm_cut_off = high_pass_cut_off/sampling_rate
    b, a = butter(high_pass_filter_order, norm_cut_off, 'high')
    filtered_signal = filtfilt(b, a, signal_to_filt)
    return filtered_signal

# perform PCA
def f_pca(multivariable_time_series, n_components):
    pca = PCA(n_components)
    X = pca.fit(multivariable_time_series).transform(multivariable_time_series)
    return X

# integration function
def f_integreate_trapezoid(y, fs=200, initial=0):
    output = cumulative_trapezoid(y, x=None, dx = 1/fs, initial = initial)
    return output

# calculate total harmonic distortion of a given signal
def f_thd(time_series, sampling_rate):
    xf, yf = f_fft(time_series, sampling_rate)

    fund_freq = xf[np.where(yf[xf<1] == np.max(yf[xf<1]))[0][0]]
    fund_freq_idx = np.where(xf==fund_freq)[0][0]

    amps = []

    for i in range(2, 8):
        idx_to_consider = int(fund_freq_idx * i)
        lower_bound = 0.95 * xf[idx_to_consider]
        upper_bound = 1.05 * xf[idx_to_consider]

        low_bound_idx = np.where(np.abs((xf - lower_bound)) == np.min(np.abs(xf - lower_bound)))[0][0]
        up_bound_idx = np.where(np.abs((xf - upper_bound)) == np.min(np.abs(xf - upper_bound)))[0][0]

        data_to_consider = yf[low_bound_idx:up_bound_idx]
        peaks, _ = find_peaks(data_to_consider)

        if len(peaks)>0:
            val = np.max(np.array(yf)[np.arange(low_bound_idx, up_bound_idx)[peaks]])
        else:
            val = np.nan
        amps.append(val)
    
    sum_harmonics = 0
    for value in amps:
        sum_harmonics = sum_harmonics + value ** 2

    thd = (sum_harmonics**0.5) * 100 / np.max(yf[xf<1])
    return thd
# ========================================================================================================

# %%
# ========================================================================================================
# Calculate Angular Velocity from Gyro Data
# ========================================================================================================

def f_get_angl_vel_components(raw_gyro, fs, cut_off_frequency = 3, filter_order = 2, n_components=3):

    sig_1 = f_low_pass_filter(raw_gyro['X'], fs, cut_off_frequency, filter_order)
    sig_2 = f_low_pass_filter(raw_gyro['Y'], fs, cut_off_frequency, filter_order)
    sig_3 = f_low_pass_filter(raw_gyro['Z'], fs, cut_off_frequency, filter_order)
    angl_vel_components = f_pca(np.array([sig_1, sig_2, sig_3]).T, n_components)
    # angl_vel_components[:,0] is the angular velocity in direction of motion
    return angl_vel_components[:,0]
# ========================================================================================================

# %%
# ========================================================================================================
# calculate angles from angular velocity (first component)
# ========================================================================================================

# calculate angles from angl vel
def f_calculate_angles(angular_velocity_data, fs, initial_angles_val, high_pass_cut_off, high_pass_filter_order):
    # step 1: integration of angl vel to angles
    angl_data = f_integreate_trapezoid(angular_velocity_data, fs, initial_angles_val)
    # step 2: filter out low-freq inegration noise from angles calculated
    angl_data = f_high_pass_filter(angl_data, fs, high_pass_cut_off, high_pass_filter_order)
    angl_data = angl_data
    return angl_data.tolist()
# ========================================================================================================

# %%
# ========================================================================================================
# Calculate number of cycles
# ========================================================================================================

# get dominant freq in a given time series
def f_get_dominant_freq(time_series, sampling_rate):
    xf, yf  = f_fft(time_series, sampling_rate)
    return float(xf[np.where(np.max(yf)==yf)])

# get peaks in angle data
def f_get_angle_peaks(time_series, min_prominence, min_spacing):
    angle_peaks_vals = []
    angle_peaks_idxs = []

    # peaks in positive side of series
    peaks_pos, _ = find_peaks(time_series, prominence=min_prominence)
    # peaks in negative side of series
    peaks_neg, _ = find_peaks(-1 * np.array(time_series), prominence=min_prominence)

    # all peaks - only one peak of opp sign is present between two consecutive peaks of same sign
    if peaks_neg[0] < peaks_pos[0]:     # if neg peak is the first peak in the data
        angle_peaks_idxs.append(peaks_neg[0])

        for idx in range(1, len(peaks_neg)):    # only consider consecutive same sign peaks that are separated by at least min_spacing
            if peaks_neg[idx] - peaks_neg[idx-1] > min_spacing:
                angle_peaks_idxs.append(peaks_neg[idx])
            
            indices_range = np.arange(peaks_neg[idx-1], peaks_neg[idx]) # all indices between the consecutive peaks
            
            dummy_arr = []
            for peak in peaks_pos:
                if peak in indices_range:
                    dummy_arr.append(peak)  # all opp sign peaks between two consecutive peaks of same sign

            if dummy_arr:  # from all opp sign peaks between two consecutive peaks of same sign, find the largest
                dummy_vals = np.array(time_series)[dummy_arr]
                val_to_consider = np.max(dummy_vals)
                idx_to_consider = dummy_arr[np.where(np.max(dummy_vals)==val_to_consider)[0][0]]

                angle_peaks_idxs.append(idx_to_consider)
        

    else:
        angle_peaks_idxs.append(peaks_pos[0])     # if pos peak is the first peak in the data

        for idx in range(1, len(peaks_pos)):    # only consider consecutive same sign peaks that are separated by at least min_spacing
            if peaks_pos[idx] - peaks_pos[idx-1] > min_spacing:
                angle_peaks_idxs.append(peaks_pos[idx])
            
            indices_range = np.arange(peaks_pos[idx-1], peaks_pos[idx]) # all indices between the consecutive peaks
            
            dummy_arr = []
            for peak in peaks_neg:
                if peak in indices_range:
                    dummy_arr.append(peak)  # all opp sign peaks between two consecutive peaks of same sign

            if dummy_arr:  # from all opp sign peaks between two consecutive peaks of same sign, find the largest
                dummy_vals = np.array(time_series)[dummy_arr]
                val_to_consider = np.min(dummy_vals)
                idx_to_consider = dummy_arr[np.where(np.min(dummy_vals)==val_to_consider)[0][0]]

                angle_peaks_idxs.append(idx_to_consider)

    return angle_peaks_vals, angle_peaks_idxs

def f_get_counts(angles_data, min_prominence=2, min_spacing=50):

    peaks_vals, peaks_idxs  = f_get_angle_peaks(angles_data, min_prominence, min_spacing)

    peaks_idxs = list(np.sort(peaks_idxs))
    pos_cycle = []
    neg_cycle = []
    # if val goes low to high --> postive swing
    for idx in range(len(peaks_idxs)-1):
        if angles_data[peaks_idxs[idx+1]] > angles_data[peaks_idxs[idx]]:
            pos_cycle.append((peaks_idxs[idx], peaks_idxs[idx+1]))

    # if val goes high to low --> negative swing
    for idx in range(len(peaks_idxs)-1):
        if angles_data[peaks_idxs[idx+1]] < angles_data[peaks_idxs[idx]]:
            neg_cycle.append((peaks_idxs[idx], peaks_idxs[idx+1]))

    # one positive + one negative swing --> one cycle of swing
    if len(pos_cycle) == len(neg_cycle):
        counts = len(pos_cycle)
    else:
        counts = max([len(pos_cycle), len(neg_cycle)])


    cycles = {}
    cycles['counts'] = counts
    cycles['positive'] = pos_cycle
    cycles['negative'] = neg_cycle
    return cycles
# ========================================================================================================

# %%
# ========================================================================================================
# Metrics from Angles Data
# ========================================================================================================
def f_get_range_of_motion(angles_data, cycles):

    pos_cycle = cycles['positive']
    neg_cycle = cycles['negative']

    range_of_motion_pos_cycles = []
    range_of_motion_neg_cycles = []

    for idx in range(len(pos_cycle)):
        range_of_motion_pos_cycles.append(np.abs(angles_data[pos_cycle[idx][1]] - angles_data[pos_cycle[idx][0]]))

    for idx in range(len(neg_cycle)):
        range_of_motion_neg_cycles.append(np.abs(angles_data[neg_cycle[idx][1]] - angles_data[neg_cycle[idx][0]]))

    range_of_motion = {}
    range_of_motion['positive cycles'] = {}
    range_of_motion['negative cycles'] = {}
    range_of_motion['cumulative'] = {}
    range_of_motion['positive cycles']['values'] = range_of_motion_pos_cycles
    range_of_motion['negative cycles']['values'] = range_of_motion_neg_cycles
    range_of_motion['positive cycles']['variance'] = np.nanvar(range_of_motion_pos_cycles)
    range_of_motion['negative cycles']['variance'] = np.nanvar(range_of_motion_neg_cycles)
    range_of_motion['positive cycles']['mean'] = np.nanmean(range_of_motion_pos_cycles)
    range_of_motion['negative cycles']['mean'] = np.nanmean(range_of_motion_neg_cycles)
    range_of_motion['cumulative']['mean'] = np.nanmean(range_of_motion_pos_cycles + range_of_motion_neg_cycles)
    range_of_motion['cumulative']['variance'] = np.nanvar(range_of_motion_pos_cycles + range_of_motion_neg_cycles)
    range_of_motion['cumulative']['max'] = np.nanmax(range_of_motion_pos_cycles + range_of_motion_neg_cycles)

    # plt.figure()
    # for idx in range(len(pos_cycle)):
    #     plt.scatter(pos_cycle[idx][0], range_of_motion['positive cycles']['values'][idx], color='r', alpha=0.3)            
    # for idx in range(len(neg_cycle)):
    #     plt.scatter(neg_cycle[idx][0], range_of_motion['negative cycles']['values'][idx], color='b', alpha=0.3)
    # plt.tight_layout()
    # plt.show()
    return range_of_motion

def f_get_swing_temporal(angles_data, cycles, fs=200):

    pos_cycle = cycles['positive']
    neg_cycle = cycles['negative']

    swing_times = {}
    swing_times['positive'] = {}
    swing_times['positive']['values'] = []
    swing_times['positive']['mean'] = []
    swing_times['positive']['variance'] = []
    swing_times['positive']['max'] = []

    swing_times['negative'] = {}
    swing_times['negative']['values'] = []
    swing_times['negative']['mean'] = []
    swing_times['negative']['variance'] = []
    swing_times['negative']['max'] = []

    swing_times['cumulative'] = {}
    swing_times['cumulative']['mean'] = []
    swing_times['cumulative']['variance'] = []
    swing_times['cumulative']['max'] = []

    for swing in pos_cycle:
        swing_times['positive']['values'].append(np.abs((swing[1]-swing[0])/fs))

    swing_times['positive']['mean'] = np.nanmean(swing_times['positive']['values'])
    swing_times['positive']['variance'] = np.nanvar(swing_times['positive']['values'])
    swing_times['positive']['max'] = np.nanmax(swing_times['positive']['values'])


    for swing in neg_cycle:
        swing_times['negative']['values'].append(np.abs((swing[1]-swing[0])/fs))

    swing_times['negative']['mean'] = np.nanmean(swing_times['negative']['values'])
    swing_times['negative']['variance'] = np.nanvar(swing_times['negative']['values'])
    swing_times['negative']['max'] = np.nanmax(swing_times['negative']['values'])


    swing_times['cumulative']['mean'] = np.nanmean(swing_times['positive']['values'] + swing_times['negative']['values'])
    swing_times['cumulative']['variance'] = np.nanvar(swing_times['positive']['values'] + swing_times['negative']['values'])
    swing_times['cumulative']['max'] = np.nanmax(swing_times['positive']['values'] + swing_times['negative']['values'])

    # plt.figure()
    # for swing in pos_cycle:
    #     plt.scatter(swing[0], np.abs(swing[1]-swing[0])/fs, color='r', alpha=0.3)          
    # for swing in neg_cycle:
    #     plt.scatter(swing[0], np.abs(swing[1]-swing[0])/fs, color='b', alpha=0.3)
    # plt.tight_layout()
    # plt.show()
    return swing_times

def f_get_swing_amplitudes(angles_data, cycles):

    pos_cycle = cycles['positive']
    neg_cycle = cycles['negative']

    swing_amplitudes = {}
    swing_amplitudes['positive'] = {}
    swing_amplitudes['positive']['values'] = []
    swing_amplitudes['positive']['mean'] = []
    swing_amplitudes['positive']['variance'] = []
    swing_amplitudes['positive']['max'] = []

    swing_amplitudes['negative'] = {}
    swing_amplitudes['negative']['values'] = []
    swing_amplitudes['negative']['mean'] = []
    swing_amplitudes['negative']['variance'] = []
    swing_amplitudes['negative']['max'] = []

    swing_amplitudes['cumulative'] = {}
    swing_amplitudes['cumulative']['mean'] = []
    swing_amplitudes['cumulative']['variance'] = []
    swing_amplitudes['cumulative']['max'] = []

    for swing in pos_cycle:
        swing_amplitudes['positive']['values'].append(np.max([np.abs(angles_data[swing[0]]), np.abs(angles_data[swing[0]])]))

    swing_amplitudes['positive']['mean'] = np.nanmean(swing_amplitudes['positive']['values'])
    swing_amplitudes['positive']['variance'] = np.nanvar(swing_amplitudes['positive']['values'])
    swing_amplitudes['positive']['max'] = np.nanmax(swing_amplitudes['positive']['values'])


    for swing in neg_cycle:
        swing_amplitudes['negative']['values'].append(np.max([np.abs(angles_data[swing[0]]), np.abs(angles_data[swing[0]])]))

    swing_amplitudes['negative']['mean'] = np.nanmean(swing_amplitudes['negative']['values'])
    swing_amplitudes['negative']['variance'] = np.nanvar(swing_amplitudes['negative']['values'])
    swing_amplitudes['negative']['max'] = np.nanmax(swing_amplitudes['negative']['values'])


    swing_amplitudes['cumulative']['mean'] = np.nanmean(swing_amplitudes['positive']['values'] + swing_amplitudes['negative']['values'])
    swing_amplitudes['cumulative']['variance'] = np.nanvar(swing_amplitudes['positive']['values'] + swing_amplitudes['negative']['values'])
    swing_amplitudes['cumulative']['max'] = np.nanmax(swing_amplitudes['positive']['values'] + swing_amplitudes['negative']['values'])
    return swing_amplitudes
# ========================================================================================================


# %%
# ========================================================================================================
# metrics from angl vel data
# ========================================================================================================
def f_get_angl_vel_peaks(angl_vel, cycles):
    angl_vel_peak_idxs = []
    angl_vel_peaks_values = []
    for cycle in cycles['positive'] + cycles['negative']:
        val_to_consider = []
        idx_to_consider = []
        idxs = np.arange(cycle[0], cycle[1])
        peaks, _ = find_peaks(np.array(angl_vel[idxs]), prominence=10)
        if len(peaks)>0:
            val_to_consider = np.max(angl_vel[idxs[peaks]])
            idx_to_consider = idxs[np.where(angl_vel[idxs]==val_to_consider)[0][0]]
            angl_vel_peak_idxs.append(idx_to_consider)

        else:
            peaks, _ = find_peaks(-1 * np.array(angl_vel[idxs]), prominence=10)
            
            if len(peaks) > 0:
                val_to_consider = np.max(angl_vel[idxs[peaks]])
                idx_to_consider = idxs[np.where(angl_vel[idxs]==val_to_consider)[0][0]]
                angl_vel_peak_idxs.append(idx_to_consider)
            else:
                angl_vel_peak_idxs.append(np.nan)

    for i in range(len(angl_vel_peak_idxs)):
        if np.isnan(angl_vel_peak_idxs[i]):
            angl_vel_peaks_values.append(np.nan)
        else:
            angl_vel_peaks_values.append(angl_vel[angl_vel_peak_idxs[i]])

    angl_vel_peaks = {}
    angl_vel_peaks['values'] = angl_vel_peaks_values
    angl_vel_peaks['idxs'] = angl_vel_peak_idxs

    return angl_vel_peaks
# ========================================================================================================

def f_extract_gyro_metrics(gyro_data, fs=50, initial_angles_val=0, high_pass_cut_off=0.3, high_pass_filter_order=2, low_pass_cut_off=10, low_pass_filter_order=2, min_prominence=2, min_spacing=10):
    angl_vel = f_get_angl_vel_components(gyro_data, fs)
    angles = f_calculate_angles(angl_vel, fs, initial_angles_val, high_pass_cut_off, high_pass_filter_order)


    cycles = f_get_counts(angles, min_prominence, min_spacing)
    angl_vel_peaks = f_get_angl_vel_peaks(angl_vel, cycles)
    
    swing_amplitudes = f_get_swing_amplitudes(angles, cycles)
    swing_times =  f_get_swing_temporal(angles, cycles, fs=200)
    range_of_motion = f_get_range_of_motion(angles, cycles)

    metrics = {}
    metrics['angles'] = {}
    metrics['angular_velocities'] = {}

    metrics['angular_velocities']['time_series'] = angl_vel
    metrics['angles']['time_series'] = angles

    metrics['angles']['swing amplitudes'] = swing_amplitudes
    metrics['angles']['swing times'] = swing_times
    metrics['angles']['range of motion'] = range_of_motion
    metrics['angles']['cycles'] = cycles

    metrics['angular_velocities']['mean'] = np.nanmean(np.abs(angl_vel_peaks['values']))
    metrics['angular_velocities']['max'] = np.nanmax(np.abs(angl_vel_peaks['values']))
    metrics['angular_velocities']['variance'] = np.nanvar(np.abs(angl_vel_peaks['values']))
    metrics['angular_velocities']['peaks'] = angl_vel_peaks
    metrics['angular_velocities']['thd'] = f_thd(angl_vel, fs)

    # plt.figure()
    # plt.plot(angl_vel)
    # plt.scatter(angl_vel_peaks['idxs'], angl_vel_peaks['values'])
    # plt.tight_layout()
    # plt.show()

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
        gyro_data = f_get_data_for_analysis(df, sensor_label, 'angular_velocity')
        metrics[sensor_label] = {}
        metrics[sensor_label]  = f_extract_gyro_metrics(gyro_data, fs=50, initial_angles_val=0, high_pass_cut_off=0.3, high_pass_filter_order=2, low_pass_cut_off=10, low_pass_filter_order=2, min_prominence=2, min_spacing=10)

    # plt.figure()
    # plt.plot(metrics[sensor_label]['angular_velocities']['time_series'])
    # plt.scatter(metrics[sensor_label]['angular_velocities']['peaks']['idxs'], metrics[sensor_label]['angular_velocities']['peaks']['values'])
    # plt.show()
