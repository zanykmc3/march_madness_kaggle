import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

df_tour = pd.read_csv("DataFiles/NCAATourneyDetailedResults.csv")
df_elo = pd.read_csv("season_elos.csv")
df_elo2018 = pd.read_csv("2018season_elos.csv")
df_seeds = pd.read_csv("DataFiles/NCAATourneySeeds.csv")
df_tour = df_tour[['Season','WTeamID','LTeamID']]
df_pom = pd.read_csv("Pom.csv")
df_pom2018 = pd.read_csv("Pom2018.csv")

def seed_to_int(seed):
    s_int = int(seed[1:3])
    return s_int

df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1)
df_seeds.head()

df_winadjo = df_pom.rename(columns={'TeamID':'WTeamID', 'AdjO':'WAdjO', 'AdjD':'WAdjD','AdjT':'WAdjT'})
df_lossadjo = df_pom.rename(columns={'TeamID':'LTeamID', 'AdjO':'LAdjO', 'AdjD':'LAdjD','AdjT':'LAdjT'})

df_winelo = df_elo.rename(columns={'team_id':'WTeamID', 'season_elo':'Welo'})
df_losselo = df_elo.rename(columns={'team_id':'LTeamID', 'season_elo':'Lelo'})

df_winseeds = df_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
df_lossseeds = df_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})
df_dummy = pd.merge(left=df_tour, right=df_winelo, how='left', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_dummy, right=df_losselo, on=['Season', 'LTeamID'])

df_concat1 = pd.merge(left=df_concat, right=df_winadjo, how='left', on=['Season', 'WTeamID'])
df_concat2 = pd.merge(left=df_concat1, right=df_lossadjo, on=['Season', 'LTeamID'])
df_concat2['ELODiff'] = df_concat2.Welo - df_concat2.Lelo
df_concat2['AdjODiff'] = df_concat2.WAdjO - df_concat2.LAdjO
df_concat2['AdjDDiff'] = df_concat2.WAdjD - df_concat2.LAdjD
df_concat2['AdjTDiff'] = df_concat2.WAdjT - df_concat2.LAdjT

df_wins = pd.DataFrame()
df_wins['AdjDDiff'] = df_concat2['AdjDDiff']
df_wins['AdjODiff'] = df_concat2['AdjODiff']
df_wins['AdjTDiff'] = df_concat2['AdjTDiff']
df_wins['ELODiff'] = df_concat2['ELODiff']
df_wins['Result'] = 1

df_losses = pd.DataFrame()
df_losses['AdjDDiff'] = -df_concat2['AdjDDiff']
df_losses['AdjODiff'] = -df_concat2['AdjODiff']
df_losses['AdjTDiff'] = -df_concat2['AdjTDiff']
df_losses['ELODiff'] = -df_concat2['ELODiff']
df_losses['Result'] = 0

df_predictions = pd.concat((df_wins, df_losses))
df_predictions.head()
df_predictions.fillna(0,inplace=True)
df_pom.fillna(0,inplace=True)
X_traina = df_predictions.ELODiff.values.reshape(-1,1)
X_trainb = df_predictions.AdjDDiff.values.reshape(-1,1)
X_trainc = df_predictions.AdjODiff.values.reshape(-1,1)
X_traind = df_predictions.AdjTDiff.values.reshape(-1,1)
X_train = np.column_stack((X_traina,X_trainb,X_trainc,X_traind))
print(X_train)
y_train = df_predictions.Result.values
X_train, y_train = shuffle(X_train, y_train)

logreg = LogisticRegression()
params = {'C': np.logspace(start=-5, stop=3, num=9)}
clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)
clf.fit(X_train, y_train)
print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))

df_sample_sub = pd.read_csv('SampleSubmissionStage2.csv')

n_test_games = len(df_sample_sub)

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))

X_test = np.zeros(shape=(n_test_games,4))
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_elo = df_elo2018[(df_elo2018.team_id == t1) & (df_elo2018.season == year)].season_elo.values[0]
    t2_elo = df_elo2018[(df_elo2018.team_id == t2) & (df_elo2018.season == year)].season_elo.values[0]

    try:
        t1_adjo = df_pom2018[(df_pom2018.TeamID == t1) & (df_pom2018.Season == year)].AdjO.values[0]
        t2_adjo = df_pom2018[(df_pom2018.TeamID == t2) & (df_pom2018.Season == year)].AdjO.values[0]
        t1_adjd = df_pom2018[(df_pom2018.TeamID == t1) & (df_pom2018.Season == year)].AdjD.values[0]
        t2_adjd = df_pom2018[(df_pom2018.TeamID == t2) & (df_pom2018.Season == year)].AdjD.values[0]
        t1_adjt = df_pom2018[(df_pom2018.TeamID == t1) & (df_pom2018.Season == year)].AdjT.values[0]
        t2_adjt = df_pom2018[(df_pom2018.TeamID == t2) & (df_pom2018.Season == year)].AdjT.values[0]
    except:
        t1_adjo = 0
        t2_adjo =0
        
    diff_elo = t1_elo - t2_elo
    X_test[ii, 0] = diff_elo
    
    diff_adjo = t1_adjo - t2_adjo
    X_test[ii, 1] = diff_adjo
    
    diff_adjd = t1_adjd - t2_adjd
    X_test[ii, 2] = diff_adjd
    
    diff_adjt = t1_adjt - t2_adjt
    X_test[ii, 3] = diff_adjt
    
preds = clf.predict_proba(X_test)[:,0]
clipped_preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.Pred = clipped_preds
df_sample_sub.to_csv('2018predictions-Final.csv', index=False)

