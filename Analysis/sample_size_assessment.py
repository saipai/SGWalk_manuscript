# %%
"""
Is the study size sufficient?

# figure 2 in the manuscript (cov and av for week four left hand sensor)
"""

# imports
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


path_to_file = r"Results/metrics_Aug_07_2023.json"
f = open(path_to_file)
metrics = json.load(f)
metrics = json.loads(metrics)
f.close()

winners = [15, 3, 11, 14, 5, 25, 17, 9, 29, 26, 18, 6]
losers = [13, 16, 20, 21, 7, 12, 9, 23, 22, 2, 4, 1]


# initialize data frame
games = ["Arctic Punch", "Fruit Ninja", "Piano Step"]
# games = ['Arctic Punch']
sensor_labels = ["left", "right"]


raw_metrics_frame_columns = [
    "Participant ID",
    "Winner",
    "ROM_L_wk1",
    "AV_L_wk1",
    "ROM_L_wk2",
    "AV_L_wk2",
    "ROM_L_wk3",
    "AV_L_wk3",
    "ROM_L_wk4",
    "AV_L_wk4",
    "ROM_R_wk1",
    "AV_R_wk1",
    "ROM_R_wk2",
    "AV_R_wk2",
    "ROM_R_wk3",
    "AV_R_wk3",
    "ROM_R_wk4",
    "AV_R_wk4",
]

dataset = {}

for game in games:
    dataset[game] = {}
    raw_df = pd.DataFrame([], columns=raw_metrics_frame_columns)

    for participant in range(1, 31):
        key = "participant0" + str(participant)
        data_vect = []
        # participant ID
        data_vect.append(participant)
        # winner or not
        if participant in winners:
            data_vect.append(0)
        else:
            data_vect.append((1))

        for limb in sensor_labels:
            if game in ["Arctic Punch", "Fruit Ninja"]:
                sensor_label = limb + " hand"
            elif game in ["Piano Step"]:
                sensor_label = limb + " leg"

            for week_num in [1, 2, 3, 4]:
                week = "week" + str(week_num)

                if metrics[game][week][key][sensor_label]:
                    data_vect.append(
                        metrics[game][week][key][sensor_label]["angles"][
                            "range of motion"
                        ]["cumulative"]["mean"]
                    )
                    data_vect.append(
                        metrics[game][week][key][sensor_label]["angular_velocities"][
                            "mean"
                        ]
                    )

                else:
                    data_vect.append(np.nan)
                    data_vect.append(np.nan)

        raw_df.loc[len(raw_df)] = data_vect
    dataset[game]["mean_vals"] = raw_df.to_dict()

    for participant in range(1, 31):
        key = "participant0" + str(participant)
        data_vect = []
        # participant ID
        data_vect.append(participant)
        # winner or not
        if participant in winners:
            data_vect.append(0)
        else:
            data_vect.append((1))

        for limb in sensor_labels:
            if game in ["Arctic Punch", "Fruit Ninja"]:
                sensor_label = limb + " hand"
            elif game in ["Piano Step"]:
                sensor_label = limb + " leg"

            for week_num in [1, 2, 3, 4]:
                week = "week" + str(week_num)

                if metrics[game][week][key][sensor_label]:
                    data_vect.append(
                        metrics[game][week][key][sensor_label]["angles"][
                            "range of motion"
                        ]["cumulative"]["variance"]
                        ** 0.5
                        / metrics[game][week][key][sensor_label]["angles"][
                            "range of motion"
                        ]["cumulative"]["mean"]
                    )
                    data_vect.append(
                        metrics[game][week][key][sensor_label]["angular_velocities"][
                            "variance"
                        ]
                        ** 0.5
                        / metrics[game][week][key][sensor_label]["angular_velocities"][
                            "mean"
                        ]
                    )
                else:
                    data_vect.append(np.nan)
                    data_vect.append(np.nan)

        raw_df.loc[len(raw_df)] = data_vect
    dataset[game]["cov_vals"] = raw_df.to_dict()

# %%
study_stats = {}
for game in dataset.keys():
    study_stats[game] = {}
    for param_type in dataset[game].keys():
        study_stats[game][param_type] = {}
        df = pd.DataFrame(dataset[game][param_type])
        df.dropna(inplace=True)
        for week in range(1, 5):
            study_stats[game][param_type]["Week_" + str(week)] = {}
            for feature in df.columns:
                if str(week) in feature:
                    study_stats[game][param_type]["Week_" + str(week)][feature] = {}
                    for sample_size in [5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
                        var_samples = []
                        for i in range(10):
                            var_samples.append(
                                df[feature].sample(sample_size, replace=False).std()
                            )
                        study_stats[game][param_type]["Week_" + str(week)][feature][
                            "sample_size_" + str(sample_size)
                        ] = var_samples

# x_ticks = [int(x.split('_')[1]) for x in study_stats[game][param_type][week][feature].keys()]

# define path to save
path_ = "/1TB/SGWalk_IntraCREATE/Codes_24072023/Results_Sept_04_2024"
if not os.path.exists(path_):
    os.makedirs(path_)
filename = os.path.join(path_, "study_stats.pkl")

with open(filename, "wb") as file:
    # save study statisics
    pickle.dump(study_stats, file)

for game in study_stats.keys():
    for param_type in study_stats[game].keys():
        for week in study_stats[game][param_type].keys():
            for feature in study_stats[game][param_type][week].keys():
                plt.figure()
                plt.title(game + "_" + param_type + "_" + week + "_" + feature)
                plt.boxplot(
                    study_stats[game][param_type][week][feature].values(),
                    patch_artist=True,
                    boxprops=dict(facecolor="lightyellow", color="orange", alpha=0.9),
                    medianprops=dict(color="orange", linewidth=1.5, alpha=0.9),
                    whiskerprops=dict(color="orange", linewidth=2, alpha=0.9),
                    capprops=dict(color="orange", linewidth=2, alpha=0.9),
                    sym="",
                )
                plt.xticks(
                    np.arange(
                        1, len(study_stats[game][param_type][week][feature].keys()) + 1
                    ),
                    [
                        int(x.split("_")[2])
                        for x in study_stats[game][param_type][week][feature].keys()
                    ],
                )

                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        path_,
                        game + "_" + param_type + "_" + week + "_" + feature + ".png",
                    )
                )
                plt.savefig(
                    os.path.join(
                        path_,
                        game + "_" + param_type + "_" + week + "_" + feature + ".pdf",
                    )
                )
                plt.close()
