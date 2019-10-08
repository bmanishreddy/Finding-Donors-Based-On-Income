import numpy as np
import pandas as pd
# from GradientBoostingClassifier.processing.validation import validate_inputs
from GradientBoostingClassifier.processing.data_management import load_pipeline
from GradientBoostingClassifier.config import config
from GradientBoostingClassifier import __version__ as _version

import logging
import json

_logger = logging.getLogger(__name__)

pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
_price_pipe = load_pipeline(file_name=pipeline_file_name)


# min max revrsre
def MinMax(x, xmin, xmax):
    num = int(x) - xmin
    den = xmax - xmin
    res = num / den

    return res


# Converter
def ObjConv(obj):
    age = MinMax(obj['age'], 17, 90)
    education = MinMax(obj['education-num'], 1, 16)
    capital = MinMax(obj['capital-gain'], 0.0, 99999.0)
    hours_per_week = MinMax(obj['hours-per-week'], 1.0, 99.0)

    # relationship = "Wife"

    if obj['marital-status'] == 'Divorced':
        marital_status_Divorced = 1
    else:
        marital_status_Divorced = 0
    if obj['marital-status'] == 'Married-AF-spouse':
        marital_status_Married_AF_spouse = 1
    else:
        marital_status_Married_AF_spouse = 0
    if obj['marital-status'] == 'Married-civ-spouse':
        marital_status_Married_civ_spouse = 1
    else:
        marital_status_Married_civ_spouse = 0
    if obj['marital-status'] == 'Married-spouse-absent':
        marital_status_Married_spouse_absent = 1
    else:
        marital_status_Married_spouse_absent = 0
    if obj['marital-status'] == 'Never-married':
        marital_status_Never_married = 1
    else:
        marital_status_Never_married = 0
    if obj['marital-status'] == 'Separated':
        marital_status_Separated = 1
    else:
        marital_status_Separated = 0
    if obj['marital-status'] == 'Widowed':
        marital_status_Widowed = 1
    else:
        marital_status_Widowed = 0
    if obj['relationship'] == 'Husband':
        relationship_Husband = 1
    else:
        relationship_Husband = 0
    if obj['relationship'] == 'Not-in-family':
        relationship_Not_in_family = 1
    else:
        relationship_Not_in_family = 0
    if obj['relationship'] == 'Other-relative':
        relationship_Other_relative = 1
    else:
        relationship_Other_relative = 0
    if obj['relationship'] == 'Own-child':
        relationship_Own_child = 1
    else:
        relationship_Own_child = 0
    if obj['relationship'] == 'Unmarried':
        relationship_Unmarried = 1
    else:
        relationship_Unmarried = 0
    if obj['relationship'] == 'Wife':
        relationship_Wife = 1
    else:
        relationship_Wife = 0
    final_res = [{
        "Unnamed: 0": 0,
        "age": age,
        "education-num": education,
        "capital-gain": capital,
        "hours-per-week": hours_per_week,

        "marital-status_ Divorced": marital_status_Divorced,
        "marital-status_ Married-AF-spouse": marital_status_Married_AF_spouse,
        "marital-status_ Married-civ-spouse": marital_status_Married_civ_spouse,
        "marital-status_ Married-spouse-absent": marital_status_Married_spouse_absent,
        "marital-status_ Never-married": marital_status_Never_married,
        "marital-status_ Separated": marital_status_Separated,
        "marital-status_ Widowed": marital_status_Widowed,

        "relationship_ Husband": relationship_Husband,
        "relationship_ Not-in-family": relationship_Not_in_family,
        "relationship_ Other-relative": relationship_Other_relative,
        "relationship_ Own-child": relationship_Own_child,
        "relationship_ Unmarried": relationship_Unmarried,
        "relationship_ Wife": relationship_Wife
    }
    ]
    return json.dumps(final_res)


def make_prediction(*, input_data) -> dict:
    """Make a prediction using the saved model pipeline."""
    # print("values in the imput data  ===",input_data)

    _logger.info(f'results from predict part : {input_data}')
    print("The type of data in ==========================", type(input_data))
    print(input_data)
    input_data = json.loads(input_data)
    input_data = ObjConv(input_data)

    print("processed data = +++++++++++++++++++++=", type(input_data))

    print(input_data)
    data = pd.read_json(input_data)

    # validated_data = validate_inputs(input_data=data)

    output = _price_pipe.predict(data[config.features])
    # output = np.exp(prediction)

    results = {'predictions': output, 'version': _version}

    _logger.info(
        f'Making predictions with model version: {_version} '

        f'Predictions: {results}')
    return results



def make_prediction_test(*, input_data) -> dict:
    """Make a prediction using the saved model pipeline."""
    # print("values in the imput data  ===",input_data)

    _logger.info(f'results from predict part : {input_data}')
    print("The type of data in ==========================", type(input_data))
    print(input_data)
    #input_data = json.loads(input_data)
    input_data = ObjConv(input_data)

    print("processed data = +++++++++++++++++++++=", type(input_data))

    print(input_data)
    data = pd.read_json(input_data)

    # validated_data = validate_inputs(input_data=data)

    output = _price_pipe.predict(data[config.features])
    # output = np.exp(prediction)

    results = {'predictions': output, 'version': _version}

    _logger.info(
        f'Making predictions with model version: {_version} '

        f'Predictions: {results}')
    return results
