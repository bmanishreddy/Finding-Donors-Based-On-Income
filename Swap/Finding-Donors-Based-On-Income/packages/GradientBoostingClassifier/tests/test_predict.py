'''import math
from GradientBoostingClassifier.predict import make_prediction
from GradientBoostingClassifier.processing.data_management import load_dataset
from GradientBoostingClassifier import __version__ as _version
import pandas as pd
import pytest
import numpy
def test_make_single_prediction():
    # Given
    test_data = load_dataset(file_name='test.csv')
    single_test_json = test_data[0:1].to_json(orient='records')
    #print("_____val_____")
    print(single_test_json)

    # When
    subject = make_prediction(input_data=single_test_json)
    #print("Printing out results -------")



    print(type(subject.get('predictions')[0]))

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], numpy.int64)
    assert math.ceil(subject.get('predictions')[0]) == 0


#test_make_single_prediction()'''