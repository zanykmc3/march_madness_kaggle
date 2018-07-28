import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

df_tour = pd.read_csv("WTourney.csv")
#df_tour = df_tour[df_tour.Season < 2014]
df_elo = pd.read_csv("2018_w_season_elos.csv")
#df_elo2018 = pd.read_csv("2018_w_season_elos.csv")
df_seeds = pd.read_csv("DataFiles/NCAATourneySeeds.csv")
df_tour = df_tour[['Season','WTeamID','LTeamID']]
df_pom = pd.read_csv("Pom.csv")
df_pom2018 = pd.read_csv("Pom2018.csv")

def seed_to_int(seed):
    #Get just the digits from the seeding. Return as int
    s_int = int(seed[1:3])
    return s_int
df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label
df_seeds.head()

##df_winadjo = df_pom.rename(columns={'TeamID':'WTeamID', 'AdjO':'WAdjO', 'AdjD':'WAdjD','AdjT':'WAdjT'})
##df_lossadjo = df_pom.rename(columns={'TeamID':'LTeamID', 'AdjO':'LAdjO', 'AdjD':'LAdjD','AdjT':'LAdjT'})
##

df_winelo = df_elo.rename(columns={'team_id':'WTeamID', 'season_elo':'Welo', 'season':'Season'})
df_losselo = df_elo.rename(columns={'team_id':'LTeamID', 'season_elo':'Lelo', 'season':'Season'})

df_dummy = pd.merge(left=df_tour, right=df_winelo, how='left', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_dummy, right=df_losselo, on=['Season', 'LTeamID'])

df_concat['ELODiff'] = df_concat.Welo - df_concat.Lelo


df_wins = pd.DataFrame()
df_wins['ELODiff'] = df_concat['ELODiff']
df_wins['Result'] = 1

df_losses = pd.DataFrame()
df_losses['ELODiff'] = -df_concat['ELODiff']
df_losses['Result'] = 0

df_predictions = pd.concat((df_wins, df_losses))
df_predictions.head()
df_predictions.fillna(0,inplace=True)
df_pom.fillna(0,inplace=True)
X_traina = df_predictions.ELODiff.values.reshape(-1,1)

y_train = df_predictions.Result.values
X_traina, y_train = shuffle(X_traina, y_train)

logreg = LogisticRegression()
params = {'C': np.logspace(start=-5, stop=3, num=9)}
clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)
clf.fit(X_traina, y_train)
print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))

df_sample_sub = pd.read_csv('WSS2.csv')

n_test_games = len(df_sample_sub)

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))

X_test = np.zeros(shape=(n_test_games,1))
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_elo = df_elo[(df_elo.team_id == t1) & (df_elo.season == year)].season_elo.values[0]
    t2_elo = df_elo[(df_elo.team_id == t2) & (df_elo.season == year)].season_elo.values[0]


    diff_elo = t1_elo - t2_elo
    X_test[ii, 0] = diff_elo
    
    
preds = clf.predict_proba(X_test)[:,1]
#preds = clf.predict_proba(X_test)[:,1]

#print(predsa)
#print(preds)

clipped_preds = np.clip(preds, 0.05, 0.95)

df_sample_sub.Pred = clipped_preds

df_sample_sub.head()
df_sample_sub.to_csv('2018WomensPrediction.csv', index=False)

