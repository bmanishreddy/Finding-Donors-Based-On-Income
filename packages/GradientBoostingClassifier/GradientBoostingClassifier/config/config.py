import os
import pathlib

import GradientBoostingClassifier

import pandas as pd

PACKAGE_ROOT = pathlib.Path(GradientBoostingClassifier.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

# data
TESTING_DATA_FILE = 'test.csv'
TRAINING_DATA_FILE = 'train.csv'
TARGET = 'SalePrice'


#numerical Variable
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
