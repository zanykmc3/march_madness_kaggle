# march_madness_kaggle
Men's and Women's College Basketball predictions for 2018 Kaggle March Madness Competition.

<h4>Data Processing and Wrangling Files:</h4>
create_elo_dataframe.py - gathers ELO data for men's and women's teams
create_pom_dataframe.py - gathers Pomeroy rankings for men's and women's teams
create_training_dataframe.py - creates input data for ML algorithms

<h4>Prediction Scripts:</h4>
basic_logistic_regression_mens.py - logistic regression for men's games
basic_logistic_regression_womens.py - logistic regression for women's games
dnn_classifier.py - deep neural net using tensorflow, used for both men's and women's games
