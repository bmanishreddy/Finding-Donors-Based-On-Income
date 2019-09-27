import os


import pathlib

import GradientBoostingClassifier

import pandas as pd

import os
#import packages

#print("path to parent ----",pathlib.Path.resolve().parent)
pd.options.display.max_rows = 10
pd.options.display.max_columns = 10

PACKAGE_ROOT = pathlib.Path(GradientBoostingClassifier.__file__).resolve().parent

DATASET_DIR = PACKAGE_ROOT / 'datasets'

# data
DATA_FILE = 'census.csv'


target = 'income'
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'

#TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'

#numerical Variable
#features=['age','hours-per-week','capital-gain','education-num','relationship','marital-status']

#numerical = ['age','hours-per-week','capital-gain','education-num']

features = ['age','education-num','capital-gain','hours-per-week','marital-status_ Divorced','marital-status_ Married-AF-spouse','marital-status_ Married-civ-spouse','marital-status_ Married-spouse-absent','marital-status_ Never-married','marital-status_ Separated','marital-status_ Widowed','relationship_ Husband','relationship_ Not-in-family','relationship_ Other-relative','relationship_ Own-child','relationship_ Unmarried','relationship_ Wife']


#features=['age','hours-per-week','capital-gain','education-num']

numerical = ['age','hours-per-week','capital-gain','education-num']


#stringfeatures = ['relationship','marital-status']

log_transform = ['capital-gain']

PIPELINE_NAME = 'Gradientboost'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output_v'