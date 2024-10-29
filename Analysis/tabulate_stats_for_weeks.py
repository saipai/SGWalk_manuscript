# %%
"""
Statistics of the movement characteristics reported in supplementary materials
"""

# imports
import os
import json
import numpy as np
import pandas as pd
from datetime import date, datetime

import warnings
warnings.filterwarnings("ignore")


path_to_file = r'Results/metrics_Aug_07_2023.json'
f = open(path_to_file)
metrics = json.load(f)
metrics = json.loads(metrics)
f.close()


winners = [15, 3, 11, 14, 5, 25, 17, 9, 29, 26, 18, 6]
losers = [13, 16, 20, 21, 7, 12, 9, 23, 22, 2, 4, 1]


# %% initialize data frame
games = ['Arctic Punch', 'Fruit Ninja', 'Piano Step']
# games = ['Arctic Punch']
sensor_labels = ['left', 'right']


raw_metrics_frame_columns = ['Participant ID', 'Winner', 
                'ROM_L_wk1', 'AV_L_wk1', 'ROM_L_wk2', 'AV_L_wk2', 'ROM_L_wk3', 'AV_L_wk3', 'ROM_L_wk4', 'AV_L_wk4',
                'ROM_R_wk1', 'AV_R_wk1', 'ROM_R_wk2', 'AV_R_wk2', 'ROM_R_wk3', 'AV_R_wk3', 'ROM_R_wk4', 'AV_R_wk4']

dataset = {}
metric_stats = {}

for game in games:
    metric_stats[game] = {}
    raw_df = pd.DataFrame([], columns=raw_metrics_frame_columns)

    for participant in range(1, 31):
        key = 'participant0'+str(participant)
        data_vect = []
        # participant ID
        data_vect.append(participant)
        # winner or not
        if participant in winners:
            data_vect.append(0)
        else:
            data_vect.append((1))

        for limb in sensor_labels:
            if game in ['Arctic Punch', 'Fruit Ninja']:
                sensor_label = limb + ' hand'
            elif game in ['Piano Step']:
                sensor_label = limb + ' leg'

            for week_num in [1, 2, 3, 4]:
                week = 'week'+str(week_num)

                if metrics[game][week][key][sensor_label]:
                    # data_vect.append(metrics[game][week][key][sensor_label]['angles']['range of motion']['cumulative']['variance']**0.5 / metrics[game][week][key][sensor_label]['angles']['range of motion']['cumulative']['mean'])
                    # data_vect.append(metrics[game][week][key][sensor_label]['angular_velocities']['variance']**0.5/metrics[game][week][key][sensor_label]['angular_velocities']['mean'])               


                    data_vect.append(metrics[game][week][key][sensor_label]['angles']['range of motion']['cumulative']['mean'])
                    data_vect.append(metrics[game][week][key][sensor_label]['angular_velocities']['mean'])               


                else:
                    data_vect.append(np.nan)
                    data_vect.append(np.nan)
            
        raw_df.loc[len(raw_df)] = data_vect

    dataset[game] = raw_df.to_dict()

    df= dataset[game]
    df = pd.DataFrame(df)
    ROM_vals = (
        df.ROM_L_wk1.values.tolist()
        + df.ROM_R_wk1.values.tolist()
        + df.ROM_L_wk2.values.tolist()
        + df.ROM_R_wk2.values.tolist()
        + df.ROM_L_wk3.values.tolist()
        + df.ROM_R_wk3.values.tolist()
        + df.ROM_L_wk4.values.tolist()
        + df.ROM_R_wk4.values.tolist()
    )

    AV_vals = (
        df.AV_L_wk1.values.tolist()
        + df.AV_R_wk1.values.tolist()
        + df.AV_L_wk2.values.tolist()
        + df.AV_R_wk2.values.tolist()
        + df.AV_L_wk3.values.tolist()
        + df.AV_R_wk3.values.tolist()
        + df.AV_L_wk4.values.tolist()
        + df.AV_R_wk4.values.tolist()
    )

    Participant_vals = df["Participant ID"].values.tolist() * 8

    Week_vals = (
        [1] * len(df["Participant ID"]) * 2
        + [2] * len(df["Participant ID"]) * 2
        + [3] * len(df["Participant ID"]) * 2
        + [4] * len(df["Participant ID"]) * 2
    )

    df_mean = pd.DataFrame([])
    df_mean["Participant"] = Participant_vals
    dummy = []
    for participant in df_mean['Participant']:
        if participant in winners:
            dummy.append(0)
        else:
            dummy.append(1)
    df_mean['Winner'] = dummy
    df_mean["Week"] = Week_vals
    df_mean["ROM"] = ROM_vals
    df_mean["AV"] = AV_vals
    df_mean = df_mean.dropna()

    raw_df = pd.DataFrame([], columns=raw_metrics_frame_columns)

    for participant in range(1, 31):
        key = 'participant0'+str(participant)
        data_vect = []
        # participant ID
        data_vect.append(participant)
        # winner or not
        if participant in winners:
            data_vect.append(0)
        else:
            data_vect.append((1))

        for limb in sensor_labels:
            if game in ['Arctic Punch', 'Fruit Ninja']:
                sensor_label = limb + ' hand'
            elif game in ['Piano Step']:
                sensor_label = limb + ' leg'

            for week_num in [1, 2, 3, 4]:
                week = 'week'+str(week_num)

                if metrics[game][week][key][sensor_label]:
                    data_vect.append(metrics[game][week][key][sensor_label]['angles']['range of motion']['cumulative']['variance']**0.5 / metrics[game][week][key][sensor_label]['angles']['range of motion']['cumulative']['mean'])
                    data_vect.append(metrics[game][week][key][sensor_label]['angular_velocities']['variance']**0.5/metrics[game][week][key][sensor_label]['angular_velocities']['mean'])               


                    # data_vect.append(metrics[game][week][key][sensor_label]['angles']['range of motion']['cumulative']['mean'])
                    # data_vect.append(metrics[game][week][key][sensor_label]['angular_velocities']['mean'])               


                else:
                    data_vect.append(np.nan)
                    data_vect.append(np.nan)
            
        raw_df.loc[len(raw_df)] = data_vect

    dataset[game] = raw_df.to_dict()

    df= dataset[game]
    df = pd.DataFrame(df)
    ROM_vals = (
        df.ROM_L_wk1.values.tolist()
        + df.ROM_R_wk1.values.tolist()
        + df.ROM_L_wk2.values.tolist()
        + df.ROM_R_wk2.values.tolist()
        + df.ROM_L_wk3.values.tolist()
        + df.ROM_R_wk3.values.tolist()
        + df.ROM_L_wk4.values.tolist()
        + df.ROM_R_wk4.values.tolist()
    )

    AV_vals = (
        df.AV_L_wk1.values.tolist()
        + df.AV_R_wk1.values.tolist()
        + df.AV_L_wk2.values.tolist()
        + df.AV_R_wk2.values.tolist()
        + df.AV_L_wk3.values.tolist()
        + df.AV_R_wk3.values.tolist()
        + df.AV_L_wk4.values.tolist()
        + df.AV_R_wk4.values.tolist()
    )

    Participant_vals = df["Participant ID"].values.tolist() * 8

    Week_vals = (
        [1] * len(df["Participant ID"]) * 2
        + [2] * len(df["Participant ID"]) * 2
        + [3] * len(df["Participant ID"]) * 2
        + [4] * len(df["Participant ID"]) * 2
    )

    df_cov = pd.DataFrame([])
    df_cov["Participant"] = Participant_vals
    dummy = []
    for participant in df_cov['Participant']:
        if participant in winners:
            dummy.append(0)
        else:
            dummy.append(1)
    df_cov['Winner'] = dummy
    df_cov["Week"] = Week_vals
    df_cov["ROM"] = ROM_vals
    df_cov["AV"] = AV_vals
    df_cov = df_cov.dropna()


    # df excel sheets
    # df_stats

    groups = ['Overall', 'GroupA', 'GroupB']

    for group in groups:
        if group == 'Overall':
            winner_cat = [0,1]
        elif group == 'GroupA':
            winner_cat = [0]
        elif group == 'GroupB':
            winner_cat = [1]

        df_stats = pd.DataFrame([])

        metric_types = ['ROM_mean', 'AV_mean', 'ROM_cov', 'AV_cov']
        for metric in metric_types:
            means = []
            vars = []
            for wk in range(1,5):
                if 'mean' in metric:
                    temp = df_mean
                else:
                    temp = df_cov

                if 'ROM' in metric:
                    means.append(temp[temp['Week']==wk][temp.Winner.isin(winner_cat)]['ROM'].mean())
                    vars.append(temp[temp['Week']==wk][temp.Winner.isin(winner_cat)]['ROM'].std()/temp[temp['Week']==wk][temp.Winner.isin(winner_cat)]['ROM'].mean())
                elif 'AV' in metric:
                    means.append(temp[temp['Week']==wk][temp.Winner.isin(winner_cat)]['AV'].mean())
                    vars.append(temp[temp['Week']==wk][temp.Winner.isin(winner_cat)]['AV'].std()/temp[temp['Week']==wk][temp.Winner.isin(winner_cat)]['AV'].mean())

            df_stats['Week'] = np.arange(1,5)
            df_stats[metric + '_mean'] = means
            df_stats[metric + '_var'] = vars

        # print(df_stats)

        metric_stats[game][group] = df_stats



print(metric_stats.keys())
for key in metric_stats.keys():
    print(metric_stats[key].keys())



current_date = datetime.now().strftime('%Y-%m-%d')

for game in games:
    # Define the file path with the current date
    file_path_with_date = f'Metrics_{game}' + f'_{current_date}.xlsx'
    print(game)
    for group in metric_stats[game].keys():
        print(group)
        print(metric_stats[game][group])
        if os.path.isfile(file_path_with_date):
        # Write the DataFrame to existing Excel file with the current date in the file name
            with pd.ExcelWriter(file_path_with_date, engine='openpyxl', mode='a', if_sheet_exists = 'replace') as writer:
                metric_stats[game][group].to_excel(writer, index=False, sheet_name=group)
        else:
        # Write the DataFrame to a new Excel file with the current date in the file name
            with pd.ExcelWriter(file_path_with_date, engine='openpyxl', mode='w') as writer:
                metric_stats[game][group].to_excel(writer, index=False, sheet_name=group)
print(" ------------------------- ")

