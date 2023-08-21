

# Project
Australia Rain Classification (Data Source: [Here](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package))

# Intro #
This project includes
- [x] Exploratory Analysis
  - Normality check
  - Cardinality Check
  - Target variable class imbalance
  - KDE for preprocess steps
- [x] Preprocessing
  - Null values treatment using Mean, Median, Mode
  - Created Dummies
  - Label Encoding 
- [x] Feature Engineering
  - VIF
  - Pearson correlation
- [x] Predictive Modelling
  - Benchmark model (Logistic Regression with 200 random iterations)
  - All classifier implementation
  - GridSearchCV for finetunning
- [x] Performance Metric
  - F1_score evaluation
- [x] Pipelines
  - for categorical variables, One hot encoding and creating dummies
  - for numeric variable, replace by median and mean, and Scaling
  - Column Pipelined with XGB classifier and GridCV
- [x] Oversampling using SMOTE
- [x] Undersampling using RandomUnderSampler
- [x] After, Removing all NaN, classification using XGB best params and Undersampling.



# Technologies used #

* Python 3.11
* Sklearn

# Steps to install #
* clone this repo
* install specific library
* start running the instructions from two notebooks.