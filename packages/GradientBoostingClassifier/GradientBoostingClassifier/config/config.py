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


TARGET = 'income'
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'

#TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'

#numerical Variable
FEATURES=['age','hours-per-week','capital-gain','education-num','relationship','marital-status','income']

numerical = ['age','hours-per-week','capital-gain','education-num']

log_transform = ['capital-gain']


PIPELINE_NAME = 'Gradientboost'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output_v'