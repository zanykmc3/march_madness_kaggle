### Data Manipulation & Processing ###
import numpy as np
import pandas as pd
### Logisitic Regression & ML Libraries ###
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
### Graphing ###
import matplotlib.pyplot as plt
import pylab
from operator import add

# Load Data
df_seeds = pd.read_csv('DataFiles/NCAATourneySeeds.csv')
df_tour = pd.read_csv( 'DataFiles/NCAATourneyCompactResults.csv')
df_conf = pd.read_csv('DataFiles/TeamConferences.csv')
df_regseas = pd.read_csv('DataFiles/RegularSeasonCompactResults.csv')
df_massey = pd.read_csv('MasseyOrdinals.csv')
df_elo = pd.read_csv("1985result_elos.csv")
df_rs = pd.read_csv('WRegSeasDetailed.csv')

# Create Dicts For Computation
win_dict = {}
conf_win_dict = {}
conf_loss_dict = {}
rpi_dict = {}
stat_dict = {}
mas_dict = {}
def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))

def create_win_dict(df):
    for index, row in df.iterrows():
        win_dict[row['WTeamID']] = {}
    for index, row in df.iterrows():
        win_dict[row['WTeamID']][row['Season']] = 0
    for index, row in df.iterrows():
        win_dict[row['WTeamID']][row['Season']] = win_dict[row['WTeamID']][row['Season']]  + 1

def create_conf_win(df):
    for index, row in df.iterrows():
        conf_win_dict[row['WinConf']] = {}
        conf_loss_dict[row['LossConf']] = {}
    for index, row in df.iterrows():
        conf_win_dict[row['WinConf']][row['Season']] = 0
        conf_loss_dict[row['LossConf']][row['Season']] = 0
    for index, row in df.iterrows():
        conf_win_dict[row['WinConf']][row['Season']] = conf_win_dict[row['WinConf']][row['Season']]  + 1
        conf_loss_dict[row['LossConf']][row['Season']] = conf_loss_dict[row['LossConf']][row['Season']]  + 1

def seed_to_int(seed):
    s_int = int(seed[1:3])
    return s_int

def create_mas_dict(df):
    for index, row in df.iterrows():
        mas_dict[row['TeamID']] = {}
    for index, row in df.iterrows():
        mas_dict[row['TeamID']][row['Season']] = {}
    for index, row in df.iterrows():
        mas_dict[row['TeamID']][row['Season']][row['RankingDayNum']] = row['OrdinalRank']

def create_rpi_dict(df):
    for index, row in df.iterrows():
        rpi_dict[row['TeamID']] = {}
    for index, row in df.iterrows():
        rpi_dict[row['TeamID']][row['Season']] = {}
    for index, row in df.iterrows():
        rpi_dict[row['TeamID']][row['Season']][row['RankingDayNum']] = row['OrdinalRank']
        
def create_stat_dict(df):
    for index, row in df.iterrows():
        stat_dict[row['WTeamID']] = {}
        stat_dict[row['LTeamID']] = {}
    for index, row in df.iterrows():
        stat_dict[row['WTeamID']][row['Season']] = {}
        stat_dict[row['LTeamID']][row['Season']] = {}
    for index, row in df.iterrows():
        # wins, losses, pf, pa, 
        stat_dict[row['WTeamID']][row['Season']][row['DayNum']] = np.array([1, 0, row['WScore'], row['LScore'],row['WAst'],row['WTO'],row['WOR'],row['WDR'],row['WFGM'],row['WFGA'],row['WFGM3'],row['WFGA3'],row['WFTM'],row['WFTA'],row['WBlk'],row['WPF']])
        stat_dict[row['LTeamID']][row['Season']][row['DayNum']] = np.array([0, 1, row['LScore'], row['WScore'],row['LAst'],row['LTO'],row['LOR'],row['LDR'],row['LFGM'],row['LFGA'],row['LFGM3'],row['LFGA3'],row['LFTM'],row['LFTA'],row['LBlk'],row['LPF']])
        
create_stat_dict(df_rs)

# Take Each Teams Pre-Game Averages On All Numerical Statistical Categories And Train On Wins And Losses
# Add all previous stats together to create average
df_rs = df_rs[['Season','DayNum','WTeamID','LTeamID']]
df_rs['WinDiff'] = 0
df_rs['LossDiff'] = 0                                                                         
df_rs['PFDiff'] = 0
df_rs['PADiff'] = 0
df_rs['AstDiff'] = 0
df_rs['TODiff'] = 0
df_rs['ORDiff'] = 0
df_rs['DRDiff'] = 0
df_rs['FGMDiff'] = 0
df_rs['FGADiff'] = 0                                                                         
df_rs['FGM3Diff'] = 0
df_rs['FGA3Diff'] = 0
df_rs['FTMDiff'] = 0
df_rs['FTADiff'] = 0
df_rs['BlkDiff'] = 0
df_rs['PF2Diff'] = 0
df_rs['RPIDiff'] = 0
df_rs['MASDiff'] = 0
df_rs['ELODiff'] = 0
for index, row in df_rs.iterrows():
    # iterate through dict
    average_winteam = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    average_lossteam = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    
    for key_win, value_win in stat_dict[row['WTeamID']][row['Season']].items():
        if (key_win < row['DayNum']):
            average_winteam = average_winteam + value_win
    for key_loss, value_loss in stat_dict[row['LTeamID']][row['Season']].items():
        if (key_loss < row['DayNum']):
            average_lossteam = average_lossteam + value_loss
            
    wrpi = 0
    lrpi = 0
    wmas=0
    lmas=0
    try:
        for key_wrpi, value_wrpi in rpi_dict[row['WTeamID']][row['Season']].items():
            if(key_wrpi <= row['DayNum']):
                wrpi = value_wrpi
    except:
        pass
    try:
        for key_lrpi, value_lrpi in rpi_dict[row['LTeamID']][row['Season']].items():
            if(key_lrpi <= row['DayNum']):
                lrpi = value_lrpi
    except:
        pass
        
    try:
        for key_wmas, value_wmas in mas_dict[row['WTeamID']][row['Season']].items():
            if(key_wmas <= row['DayNum']):
                wmas = value_wmas
    except:
        pass
    try:
        for key_lmas, value_lmas in mas_dict[row['LTeamID']][row['Season']].items():
            if(key_lmas <= row['DayNum']):
                lmas = value_lmas
    except:
        pass
    wtg = (average_winteam[0]+average_winteam[1])
    ltg = (average_lossteam[0]+average_lossteam[1])
    if(wtg==0):
        wtg = 1.0
    if(ltg==0):
        ltg = 1.0
        
    average_winteam = np.append(average_winteam[0:2], average_winteam[2:16]/wtg)
    average_lossteam =  np.append(average_lossteam[0:2], average_lossteam[2:16]/ltg)
    stat_diff = average_winteam - average_lossteam
    rpi_diff = wrpi - lrpi
    mas_diff = wmas - lmas
    df_rs.set_value(index,'WinDiff', stat_diff[0])
    df_rs.set_value(index,'LossDiff', stat_diff[1])                                                                          
    df_rs.set_value(index,'PFDiff', stat_diff[2])
    df_rs.set_value(index,'PADiff',  stat_diff[3])
    
    df_rs.set_value(index,'AstDiff', stat_diff[4])
    df_rs.set_value(index,'TODiff',  stat_diff[5])                                                                          
    df_rs.set_value(index,'ORDiff',  stat_diff[6])
    df_rs.set_value(index,'DRDiff',  stat_diff[7])
    
    df_rs.set_value(index,'FGMDiff',  stat_diff[8])
    df_rs.set_value(index,'FGADiff',  stat_diff[9])                                                                          
    df_rs.set_value(index,'FGM3Diff',  stat_diff[10])
    df_rs.set_value(index,'FGA3Diff',  stat_diff[11])

    df_rs.set_value(index,'FTMDiff',  stat_diff[12])
    df_rs.set_value(index,'FTADiff',  stat_diff[13])                                                                          
    df_rs.set_value(index,'BlkDiff',  stat_diff[14])
    df_rs.set_value(index,'PF2Diff',  stat_diff[15]) 

df_rs = df_rs[df_rs.DayNum > 80]
# Create Win and Loss DataFrames for Training
df_wins = pd.DataFrame()
df_wins['WinDiff'] = df_rs['WinDiff']
df_wins['LossDiff'] = df_rs['LossDiff']                                                                        
df_wins['PFDiff'] = df_rs['PFDiff']
df_wins['PADiff'] = df_rs['PADiff']
df_wins['AstDiff'] = df_rs['AstDiff']
df_wins['TODiff'] = df_rs['TODiff']
df_wins['ORDiff'] = df_rs['ORDiff']
df_wins['DRDiff'] = df_rs['DRDiff']
df_wins['FGMDiff'] = df_rs['FGMDiff']
df_wins['FGADiff'] = df_rs['FGADiff']                                                                        
df_wins['FGM3Diff'] = df_rs['FGM3Diff']
df_wins['FGA3Diff'] = df_rs['FGA3Diff']
df_wins['FTMDiff'] = df_rs['FTMDiff']
df_wins['FTADiff'] = df_rs['FTADiff']
df_wins['BlkDiff'] = df_rs['BlkDiff']
df_wins['PF2Diff'] = df_rs['PF2Diff']
df_wins['Result'] = 1

df_loss = pd.DataFrame()
df_loss['WinDiff'] = -df_rs['WinDiff']
df_loss['LossDiff'] = -df_rs['LossDiff']                                                                        
df_loss['PFDiff'] = -df_rs['PFDiff']
df_loss['PADiff'] = -df_rs['PADiff']
df_loss['AstDiff'] = -df_rs['AstDiff']
df_loss['TODiff'] = -df_rs['TODiff']
df_loss['ORDiff'] = -df_rs['ORDiff']
df_loss['DRDiff'] = -df_rs['DRDiff']
df_loss['FGMDiff'] = -df_rs['FGMDiff']
df_loss['FGADiff'] = -df_rs['FGADiff']                                                                        
df_loss['FGM3Diff'] = -df_rs['FGM3Diff']
df_loss['FGA3Diff'] = -df_rs['FGA3Diff']
df_loss['FTMDiff'] = -df_rs['FTMDiff']
df_loss['FTADiff'] = -df_rs['FTADiff']
df_loss['BlkDiff'] = -df_rs['BlkDiff']
df_loss['PF2Diff'] = -df_rs['PF2Diff']
df_loss['Result'] = 0

df_predictions = pd.concat((df_wins, df_loss))
df_predictions.to_csv('w_game_data.csv',index=False)

