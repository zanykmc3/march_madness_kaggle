import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

game_dataframe = pd.read_csv("w_game_data.csv")
#game_dataframe = pd.read_csv("linear_game_data.csv")
game_dataframe = game_dataframe.reindex(np.random.permutation(game_dataframe.index))
regseas_dataframe = pd.read_csv("WRegSeasDetailed.csv")
submission_dataframe = pd.read_csv("WSS2.csv")
sub_dataframe = pd.read_csv("test_input.csv")

stat_dict = {}

def preprocess_features(game_dataframe):
  selected_features = game_dataframe[
    ["PADiff",
     "TODiff",
     "ORDiff",
     "FGA3Diff",
     "WinDiff",
     "AstDiff",
     "LossDiff",
     "PFDiff",
     "DRDiff",
     "FGMDiff",
     "FGADiff",
     "FGM3Diff",
     "FTMDiff",
     "FTADiff",
     "BlkDiff",
     "PF2Diff"]]
  
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  return processed_features

def construct_feature_columns(input_features):
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

def preprocess_targets(game_dataframe):
  output_targets = pd.DataFrame()
  output_targets["Result"] =  game_dataframe["Result"]
  return output_targets

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))

def create_stat_dict(df):
    for index, row in df.iterrows():
        stat_dict[row['WTeamID']] = {}
        stat_dict[row['LTeamID']] = {}
    for index, row in df.iterrows():
        stat_dict[row['WTeamID']][row['Season']] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        stat_dict[row['LTeamID']][row['Season']] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    for index, row in df.iterrows():
        # wins, losses, pf, pa, 
        stat_dict[row['WTeamID']][row['Season']] = stat_dict[row['WTeamID']][row['Season']] + np.array([1, 0, row['WScore'], row['LScore'],row['WAst'],row['WTO'],row['WOR'],row['WDR'],row['WFGM'],row['WFGA'],row['WFGM3'],row['WFGA3'],row['WFTM'],row['WFTA'],row['WBlk'],row['WPF']])
        stat_dict[row['LTeamID']][row['Season']] = stat_dict[row['LTeamID']][row['Season']] + np.array([0, 1, row['LScore'], row['WScore'],row['LAst'],row['LTO'],row['LOR'],row['LDR'],row['LFGM'],row['LFGA'],row['LFGM3'],row['LFGA3'],row['LFTM'],row['LFTA'],row['LBlk'],row['LPF']])

def create_submission_dataframe(df_rs):
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

  for index, row in df_rs.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    # Summing
    wtg = stat_dict[t1][year][0]+stat_dict[t1][year][1]
    ltg = stat_dict[t2][year][0]+stat_dict[t2][year][1]
    average_t1 = stat_dict[t1][year]
    average_t2 = stat_dict[t2][year]
    average_t1 = np.append(average_t1[0:2], average_t1[2:16]/wtg)
    average_t2 =  np.append(average_t2[0:2], average_t2[2:16]/ltg)
    stat_diff = average_t1 - average_t2
    
    # Set values
    df_rs.set_value(index,'WinDiff', stat_diff[0])
    df_rs.set_value(index,'LossDiff', stat_diff[1])                                                                          
    df_rs.set_value(index,'PFDiff', stat_diff[2])
    df_rs.set_value(index,'PADiff', stat_diff[3])                                                                          
    df_rs.set_value(index,'AstDiff', stat_diff[4])
    df_rs.set_value(index,'TODiff', stat_diff[5])                                                                          
    df_rs.set_value(index,'ORDiff', stat_diff[6])
    df_rs.set_value(index,'DRDiff', stat_diff[7])                                                                           
    df_rs.set_value(index,'FGMDiff', stat_diff[8])
    df_rs.set_value(index,'FGADiff', stat_diff[9])                                                                          
    df_rs.set_value(index,'FGM3Diff', stat_diff[10])
    df_rs.set_value(index,'FGA3Diff', stat_diff[11])
    df_rs.set_value(index,'FTMDiff', stat_diff[12])
    df_rs.set_value(index,'FTADiff', stat_diff[13])
    df_rs.set_value(index,'BlkDiff', stat_diff[14])
    df_rs.set_value(index,'PF2Diff', stat_diff[15])
    

  df_rs['Result'] = df_rs['Pred']
  df_rs.drop(labels='Pred',inplace=True,axis=1)
  #df_rs = df_rs[['ID','Result','WinDiff','LossDiff','PFDiff','PADiff','AstDiff','TODiff','ORDiff','DRDiff']]
  return df_rs

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                             
 
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified
    if shuffle:
      ds = ds.shuffle(10000)
      
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_nn_classification_model(
    learning_rate,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):

  periods = 10
  steps_per_period = steps / periods
  
  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  dnn_classifier = tf.estimator.DNNClassifier(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units
  )
  
  # Create input functions
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["Result"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["Result"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["Result"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  print ("Training model...")
  training_log_losses = []
  validation_log_losses = []
  #periods
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    dnn_classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = dnn_classifier.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['probabilities'][1] for item in training_predictions])
    
    validation_predictions = dnn_classifier.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['probabilities'][1] for item in validation_predictions])
    
    training_log_loss = metrics.log_loss(training_targets, training_predictions)
    validation_log_loss = metrics.log_loss(validation_targets, validation_predictions)
    
    # Occasionally print the current loss.
    print("Logloss:")
    print (period, training_log_loss)
    print (period, validation_log_loss)
    
    # Add the loss metrics from this period to our list.
    training_log_losses.append(training_log_loss)
    validation_log_losses.append(validation_log_loss)

  print( "Model training finished.")

  return dnn_classifier

# Find and Create Data for Submission Dataframe Features
# Creation of Dataframe/CSV
create_stat_dict(regseas_dataframe)
sub_dataframe = create_submission_dataframe(submission_dataframe)
sub_dataframe.to_csv("w_test_input.csv",index=False)

# Choose the first 12000 (out of 17000) examples for training.
training_examples = preprocess_features(game_dataframe.head(70000))
training_targets = preprocess_targets(game_dataframe.head(70000))

# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = preprocess_features(game_dataframe.tail(5000))
validation_targets = preprocess_targets(game_dataframe.tail(5000))

# Learning rate 0.01
dnn_classifier = train_nn_classification_model(
    learning_rate=0.05,
    steps=500,
    batch_size=10,
    hidden_units=[32,64,128,64,32,16],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

test_input = preprocess_features(sub_dataframe)
test_targets = preprocess_targets(sub_dataframe)
predict_test_input_fn = lambda: my_input_fn(test_input, 
                                                    test_targets["Result"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

test_predictions = dnn_classifier.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item['probabilities'][1] for item in test_predictions])
  
sub_dataframe['Result'] = test_predictions.reshape(-1,1)
sub_dataframe['Pred'] = sub_dataframe['Result']

submit_dataframe = sub_dataframe[['ID','Pred']]
submit_dataframe.to_csv('2018WomensPrediction2.csv', index=False)
