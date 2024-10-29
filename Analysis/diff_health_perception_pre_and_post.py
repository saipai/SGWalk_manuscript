# %%
"""
This file will produce the information reported in Table 1 and Table 3 of the manuscript
"""

# imports
import os
import json
import numpy as np
import pandas as pd
from datetime import date, datetime

# define path to save
path_ = "/1TB/SGWalk_IntraCREATE/Codes_24072023/Results_Oct_29_2024"
if not os.path.exists(path_):
    os.makedirs(path_)

path_to_file = r'Results/metrics_Aug_07_2023.json'
f = open(path_to_file)
metrics = json.load(f)
metrics = json.loads(metrics)
f.close()

path_to_file = '/1TB/SGWalk_IntraCREATE/Pilot_Data/IntraCreate WP4 Data-Cleaned.xlsx'
questionnaire_data = pd.read_excel(path_to_file)

winners = [15, 3, 11, 14, 5, 25, 17, 9, 29, 26, 18, 6]
losers = [13, 16, 20, 21, 7, 12, 9, 23, 22, 2, 4, 1]


features_of_interest = ["phy_health", "emo_health", "social_health", "psy_health", "Int_exc", "Int_game", "Int_wearable", "Int_app", "PU"]

dummy = []
for participant in questionnaire_data['ID']:
    if int(participant) in winners:
        dummy.append(0)
    else:
        dummy.append(1)

df = pd.DataFrame([])
df['ID'] = questionnaire_data['ID']
df['Group']  = questionnaire_data['Group']
df['is_winner'] = dummy
for feature in features_of_interest:
    dummy = feature + '_pre'
    df[dummy] = questionnaire_data[dummy]
    dummy = feature + '_post'
    df[dummy] = questionnaire_data[dummy]

df = df[df['ID']<=30]
df = df.replace({' ': np.nan})
df = df.dropna()

evaluations = {}

evaluations['winners_with_peers'] = {}
evaluations['winners_with_HC'] = {}
evaluations['losers_with_peers'] = {}
evaluations['losers_with_HC'] = {}

evaluations['winners_with_peers']['pre'] = {}
evaluations['winners_with_HC']['pre'] = {}
evaluations['losers_with_peers']['pre'] = {}
evaluations['losers_with_HC']['pre'] = {}

evaluations['winners_with_peers']['post'] = {}
evaluations['winners_with_HC']['post'] = {}
evaluations['losers_with_peers']['post'] = {}
evaluations['losers_with_HC']['post'] = {}

for feature in features_of_interest:
    dummy = feature + '_pre'
    evaluations['winners_with_peers']['pre'][feature] = df[df['is_winner']==0][df['Group']=='Peer'][dummy].mean()
    evaluations['winners_with_HC']['pre'][feature] = df[df['is_winner']==0][df['Group']=='HC'][dummy].mean()
    evaluations['losers_with_peers']['pre'][feature] = df[df['is_winner']==1][df['Group']=='Peer'][dummy].mean()
    evaluations['losers_with_HC']['pre'][feature] = df[df['is_winner']==1][df['Group']=='Peer'][dummy].mean()
    dummy = feature + '_post'
    evaluations['winners_with_peers']['post'][feature] = df[df['is_winner']==0][df['Group']=='Peer'][dummy].mean()
    evaluations['winners_with_HC']['post'][feature] = df[df['is_winner']==0][df['Group']=='HC'][dummy].mean()
    evaluations['losers_with_peers']['post'][feature] = df[df['is_winner']==1][df['Group']=='Peer'][dummy].mean()
    evaluations['losers_with_HC']['post'][feature] = df[df['is_winner']==1][df['Group']=='Peer'][dummy].mean()


for metric in ['phy_health', 'emo_health', 'social_health', 'psy_health', 'Int_exc', 'Int_game', 'Int_wearable', 'Int_app', 'PU']:
    print(metric, ':', str((df[metric+'_post'].mean() - df[metric+'_pre'].mean()) *100 / df[metric+'_pre'].mean()))

for condition in ['winner', 'not winner', 'all']:
    to_print = pd.DataFrame([], columns=["feature", "mean_change_%", "se_change_%"])
    mean_vals = []
    se_vals = []
    if condition == "winner":
        for feature in features_of_interest:
            mean_vals.append((np.mean((df[df['is_winner']==0][feature+'_post'] - df[df['is_winner']==0][feature+'_pre']) * 100 / df[df['is_winner']==0][feature+'_pre'])))
            se_vals.append((np.std((df[df['is_winner']==0][feature+'_post'] - df[df['is_winner']==0][feature+'_pre']) * 100 / df[df['is_winner']==0][feature+'_pre'])/np.sqrt(len(winners))))
        to_print["feature"] = features_of_interest
        to_print["mean_change_%"] = mean_vals
        to_print["se_change_%"] = se_vals
        to_print.to_excel(path_+"/Questionnaire_winner.xlsx")

    elif condition == "not winner":
        for feature in features_of_interest:
            mean_vals.append((np.mean((df[df['is_winner']==1][feature+'_post'] - df[df['is_winner']==1][feature+'_pre']) * 100 / df[df['is_winner']==1][feature+'_pre'])))
            se_vals.append((np.std((df[df['is_winner']==1][feature+'_post'] - df[df['is_winner']==1][feature+'_pre']) * 100 / df[df['is_winner']==1][feature+'_pre'])/np.sqrt(len(losers))))
        to_print["feature"] = features_of_interest
        to_print["mean_change_%"] = mean_vals
        to_print["se_change_%"] = se_vals
        to_print.to_excel(path_+"/Questionnaire_not_winner.xlsx")
    
    elif condition == "all":
        for feature in features_of_interest:
            mean_vals.append((np.mean((df[feature+'_post'] - df[feature+'_pre']) * 100 / df[feature+'_pre'])))
            se_vals.append((np.std((df[feature+'_post'] - df[feature+'_pre']) * 100 / df[feature+'_pre'])/np.sqrt(len(df))))
        to_print["feature"] = features_of_interest
        to_print["mean_change_%"] = mean_vals
        to_print["se_change_%"] = se_vals
        to_print.to_excel(path_+"/Questionnaire_all.xlsx")


for condition in ['winner', 'not winner', 'all']:
    to_print = pd.DataFrame([], columns=["feature", "mean_change_%", "se_change_%"])
    mean_vals = []
    se_vals = []
    if condition == "winner":
        for feature in features_of_interest:
            mean_vals.append(np.mean(df[df['is_winner']==0][feature+'_post'] - df[df['is_winner']==0][feature+'_pre']))
            se_vals.append((np.std(df[df['is_winner']==0][feature+'_post'] - df[df['is_winner']==0][feature+'_pre']))/np.sqrt(len(losers)))
        to_print["feature"] = features_of_interest
        to_print["mean_change_%"] = mean_vals
        to_print["se_change_%"] = se_vals
        to_print.to_excel(path_+"/Questionnaire_winner_abs.xlsx")

    elif condition == "not winner":
        for feature in features_of_interest:
            mean_vals.append(np.mean(df[df['is_winner']==1][feature+'_post'] - df[df['is_winner']==1][feature+'_pre']))
            se_vals.append((np.std(df[df['is_winner']==1][feature+'_post'] - df[df['is_winner']==1][feature+'_pre']))/np.sqrt(len(losers)))
        to_print["feature"] = features_of_interest
        to_print["mean_change_%"] = mean_vals
        to_print["se_change_%"] = se_vals
        to_print.to_excel(path_+"/Questionnaire_not_winner_abs.xlsx")
    
    elif condition == "all":
        for feature in features_of_interest:
            mean_vals.append(np.mean(df[feature+'_post'] - df[feature+'_pre']))
            se_vals.append((np.std(df[feature+'_post'] - df[feature+'_pre']))/np.sqrt(len(df)))
        to_print["feature"] = features_of_interest
        to_print["mean_change_%"] = mean_vals
        to_print["se_change_%"] = se_vals
        to_print.to_excel(path_+"/Questionnaire_all_abs.xlsx")
print(' ')