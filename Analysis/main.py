"""
Main file to calculate following metrics:
The output is a tuple with following in order:
1) Counts
2) Max total acceleration
3) Absolute max range of motion

for week 1-4, participants 01-10, games - 'Artic Punch', 'Fruit Ninja' and 'Piano Step'
"""

# %% imports
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date

from scipy.fft import rfft, rfftfreq
from scipy.signal import welch
from scipy.signal import find_peaks

from read_data import f_get_data_for_analysis
from extractmetrics_25072023 import f_extract_metrics

if __name__== '__main__':
    # Opening raw data file
    format = '.csv'
    fs = 50
    
    weeks = []
    for i in range(1, 5):
    # for i in range(2, 3):
        weeks.append('week' + str(i))

    games = ['Arctic Punch', 'Fruit Ninja', 'Piano Step']

    participants = []
    for i in range(1, 31):
        participants.append('participant0' + str(i))

    # sensor_labels = ['waist', 'left hand', 'right hand', 'left leg', 'right leg']

    directory = '/1TB/SGWalk_IntraCREATE/Pilot_Data/CSV Data (Rename)/'

    metrics = {}

    for game in games:
    # for game in ['Fruit Ninja']:
        metrics[game] = {}
        for week in weeks:
            metrics[game][week] = {}
            for participant in participants:
            # for participant in ['participant012']:
                metrics[game][week][participant] = {}
                if game == 'Arctic Punch':
                    sensor_labels = ['left hand', 'right hand']
                elif game == 'Fruit Ninja':
                    sensor_labels = ['left hand', 'right hand']
                elif game == 'Piano Step':
                    sensor_labels = ['left leg', 'right leg']
                
                for sensor_label in sensor_labels:
                    metrics[game][week][participant][sensor_label] = {}
                    file_name = directory + week + '_' + participant + '_' + game + format
                    if os.path.isfile(file_name):
                        f = open(file_name)
                        df = pd.read_csv(file_name)
                        f.close()  
                        print(game + ' ' + week + ' ' + participant + ' ' + sensor_label + ' ----- ')
                        print('')
                        if sensor_label in set(df.Label.values.tolist()):
                            metrics[game][week][participant][sensor_label] = f_extract_metrics(df, fs, sensor_label)
                        else:
                            print(file_name + ' ----' + sensor_label + ' doesnt exist')
                    else:
                        print(file_name + ' doesnt exist')
                        metrics[game][week][participant][sensor_label]= []

    current_dir = os.getcwd()
    
    path_to_save = current_dir + '/Results'

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    today = date.today()
    d4 = today.strftime("%b_%d_%Y")
    path_file_saved = path_to_save + '/' + 'metrics' + '_' + d4 + ".json"


    # Extend the JSONEncoder class
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    metrics = json.dumps(metrics, cls=NpEncoder)    

    with open(path_file_saved, "w+") as f:
        json.dump(metrics, f)





    # df = pd.DataFrame([])    
    # for game in games:
    #         for week in weeks:
    #             for participant in participants:
    #                 data = []
    #                 col_names = []
    #                 data.append(participant)
    #                 col_names.append('Participants')
    #                 for sensor_label in sensor_labels:
    #                     for metric in metrics[game][week][participant][sensor_label].keys():
    #                         data.append(metrics[game][week][participant][sensor_label][metric])
    #                         if sensor_label+metric not in col_names:
    #                             col_names.append(sensor_label+'_'+metric)
    #                 dummy_dict = {}
    #                 for i in range(len(col_names)):
    #                     dummy_dict[col_names[i]] = data[i]
    #                 df = df.append(dummy_dict, ignore_index=True)

    print('Analysis complete. Results are saved to file : ',  path_file_saved)
    print('')