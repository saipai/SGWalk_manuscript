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
import pandas as pd
import matplotlib.pyplot as plt

from read_data import f_get_data_for_analysis
from extract_gyro_metrics_24072023 import f_extract_gyro_metrics
from extract_acceleration_metrics_25072023 import f_extract_acceleration_metrics


def f_extract_metrics(df, fs, sensor_label, filter_accel='YES', min_accel_prominence=0.2, initial_angles_val=0, high_pass_cut_off=0.1, high_pass_filter_order=2, low_pass_cut_off=10, low_pass_filter_order=2, min_prominence=10, min_spacing=10):
    metrics = {}

    # angular velocity and angles
    gyro_data = f_get_data_for_analysis(df, sensor_label, 'angular_velocity')
    try:
      metrics = f_extract_gyro_metrics(gyro_data, fs, initial_angles_val, high_pass_cut_off, high_pass_filter_order, low_pass_cut_off, low_pass_filter_order, min_prominence, min_spacing)
    except:
       metrics = None
  # acceleration
    accel_data = f_get_data_for_analysis(df, sensor_label, 'acceleration')
    # metrics['total_acceleration'] = f_extract_acceleration_metrics(accel_data, fs, filter_accel, min_accel_prominence)
    if metrics!= None:
      metrics['total_acceleration'] = f_extract_acceleration_metrics(accel_data, fs, filter_accel, min_accel_prominence)

    
    return metrics

if __name__== '__main__':
    # Opening raw data file
    directory = '/1TB/SGWalk_IntraCREATE/Pilot_Data/CSV Data (Rename)/'
    week = 'week1'
    game = 'Fruit Ninja'
    participant = 'participant01'
    sensor_label = 'right hand'
    format = '.csv'
    fs = 50
    file_name = directory + week + '_' + participant + '_' + game + format
    f = open(file_name)
    df = pd.read_csv(file_name)
    f.close()  

    # metrics = f_extract_metrics(df, fs, sensor_label, filter_accel='YES', min_accel_prominence=0.2, initial_angles_val=0, high_pass_cut_off=0.1, high_pass_filter_order=2, low_pass_cut_off=10, low_pass_filter_order=2, min_prominence=30, min_spacing=10)
    metrics = f_extract_metrics(df, fs, sensor_label)
    
    
    # current_dir = os.getcwd()

    # path_to_save = current_dir + '/Results'

    # if not os.path.exists(path_to_save):
    #     os.makedirs(path_to_save)

    # path_file_saved = path_to_save + '/' + week + '_' + participant + '_' + game  + '_' + limb + '_' + data_type +".json"

    # with open(path_file_saved, "w+") as f:
    #     json.dump(metrics, f)
    # print('Analysis complete. Results are saved to file : ',  path_file_saved)

    current_dir = os.getcwd()

    path_to_save = current_dir + '/Results_July_11_2024'

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    font = {"family": "sans serif", "weight": "normal", "size": 8}
    plt.rc("font", **font)

    plt.figure(figsize=(3, 2))
    plt.plot(df[df['Label'] == sensor_label]['Acceleration X(g)'], alpha=0.6, linewidth =0.5, color = 'r', label='X')
    plt.plot(df[df['Label'] == sensor_label]['Acceleration Y(g)'], alpha=0.6, linewidth =0.5, color = 'b', label='Y')
    plt.plot(df[df['Label'] == sensor_label]['Acceleration Z(g)'], alpha=0.6, linewidth =0.5, color = 'g', label='Z')
    plt.xlabel('Time index')
    plt.ylabel('Acceleration (g)')
    plt.xticks([])
    plt.legend(bbox_to_anchor=(0.5, -0.37), loc='lower center', ncol=3, handleheight=2)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    path_file_saved = path_to_save + '/' + week + '_' + participant + '_' + game  + '_' + sensor_label + '_' + 'Acceleration'
    plt.savefig(path_file_saved + '.png')
    
    # plt.figure(figsize=(6, 4))
    # plt.plot(df[df['Label'] == sensor_label]['Acceleration Y(g)'])
    # plt.xlabel('Time index')
    # plt.ylabel('Acceleration - Y (g)')
    # plt.xticks([])
    # path_file_saved = path_to_save + '/' + week + '_' + participant + '_' + game  + '_' + sensor_label + '_' + 'Acceleration - Y'
    # plt.savefig(path_file_saved + '.png')

    # plt.figure(figsize=(6, 4))
    # plt.plot(df[df['Label'] == sensor_label]['Acceleration Z(g)'])
    # plt.xlabel('Time index')
    # plt.ylabel('Acceleration - Z (g)')
    # plt.xticks([])
    # path_file_saved = path_to_save + '/' + week + '_' + participant + '_' + game  + '_' + sensor_label + '_' + 'Acceleration - Z'
    # plt.savefig(path_file_saved + '.png')

    plt.figure(figsize=(3, 2))
    plt.plot(df[df['Label'] == sensor_label]['Angular velocity X(°/s)'], alpha=0.6, linewidth =0.5, color = 'r', label='X')
    plt.plot(df[df['Label'] == sensor_label]['Angular velocity Y(°/s)'], alpha=0.6, linewidth =0.5, color = 'b', label='Y')
    plt.plot(df[df['Label'] == sensor_label]['Angular velocity Z(°/s)'], alpha=0.6, linewidth =0.5, color = 'g', label='Z')
    plt.xlabel('Time index')
    plt.ylabel('Angular velocity(°/s)')
    plt.xticks([])
    plt.legend(bbox_to_anchor=(0.5, -0.37), loc='lower center', ncol=3, handleheight=2)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    path_file_saved = path_to_save + '/' + week + '_' + participant + '_' + game  + '_' + sensor_label + '_' + 'AnglVel'
    plt.savefig(path_file_saved + '.png')

    # plt.figure(figsize=(6, 4))
    # plt.plot(df[df['Label'] == sensor_label]['Angular velocity Y(°/s)'])
    # plt.xlabel('Time index')
    # plt.ylabel('Angular velocity Y(°/s)')
    # plt.xticks([])
    # path_file_saved = path_to_save + '/' + week + '_' + participant + '_' + game  + '_' + sensor_label + '_' + 'AnglVel - Y'
    # plt.savefig(path_file_saved + '.png')

    # plt.figure(figsize=(6, 4))
    # plt.plot(df[df['Label'] == sensor_label]['Angular velocity Z(°/s)'])
    # plt.xlabel('Time index')
    # plt.ylabel('Angular velocity Z(°/s)')
    # plt.xticks([])
    # path_file_saved = path_to_save + '/' + week + '_' + participant + '_' + game  + '_' + sensor_label + '_' + 'AnglVel - Z'
    # plt.savefig(path_file_saved + '.png')

    plt.show()


    print('')

# %%
