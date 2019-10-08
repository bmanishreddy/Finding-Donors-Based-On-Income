from GradientBoostingClassifier.config import config as model_config
from GradientBoostingClassifier.processing.data_management import load_dataset
from GradientBoostingClassifier import __version__ as _version
import numpy
import json
import math


def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200

'''

def test_prediction_endpoint_returns_prediction(flask_test_client):
    # Given
    # Load the test data from the regression_model package
    # This is important as it makes it harder for the test
    # data versions to get confused by not spreading it
    # across packages.
    test_data = load_dataset(file_name=model_config.TESTING_DATA_FILE)
    post_json = test_data[0:1].to_json(orient='records')





    # When
    response = flask_test_client.post('/v1/predict/GradientBoostingClassifier',
                                      json=json.loads(post_json))


    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    print("following is the response json ==",response_json)
    prediction = response_json['predictions']

    response_version = response_json['version']

    assert prediction == 'he makes less than 50 k '
    assert response_version == _version
    '''
