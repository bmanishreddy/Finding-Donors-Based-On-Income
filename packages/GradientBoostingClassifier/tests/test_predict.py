import math
from GradientBoostingClassifier.predict import make_prediction_test
from GradientBoostingClassifier.processing.data_management import load_dataset
from GradientBoostingClassifier import __version__ as _version
import pandas as pd
import pytest
import numpy


testobj = {

 "age": 35,
 "education-num": 13,
 "capital-gain": 0.0,
 "hours-per-week": 60,
 "marital-status": "Married-civ-spouse",

 "relationship": "Husband"
}





def test_make_single_prediction():
    # Given
    #test_data = load_dataset(file_name='test.csv')
    single_test_json = testobj
    #print("_____val_____")
    print(single_test_json)

    # When
    subject = make_prediction_test(input_data=single_test_json)
    #print("Printing out results -------")



    print(type(subject.get('predictions')[0]))

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], numpy.int64)
    assert math.ceil(subject.get('predictions')[0]) == 0


#test_make_single_prediction()