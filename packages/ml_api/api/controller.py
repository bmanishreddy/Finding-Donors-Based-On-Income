from flask import Blueprint, request,jsonify
from api.config import get_logger
from GradientBoostingClassifier.predict import make_prediction
import json

_logger = get_logger(logger_name=__name__)




prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'hellow world'



@prediction_app.route('/v1/predict/GradientBoostingClassifier', methods=['POST'])
def predict():
    if request.method == 'POST':
        json_data = request.get_data()
        _logger.info(f'Inputs: {json_data}')


        result = make_prediction(input_data=json_data)
        _logger.info(f'Outputs: {result}')

        predictions = int(result.get('predictions')[0])
        version = result.get('version')



        _logger.info(f'results to disp: {predictions}')

        if predictions == 1:
            predictions = "yes he makes over 50 k"
        else:
            predictions = "he makes less than 50 k "


        return jsonify({'predictions': predictions,
                        'version': version})
