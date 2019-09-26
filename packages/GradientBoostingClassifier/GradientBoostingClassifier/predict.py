import numpy as np
import pandas as pd
#from GradientBoostingClassifier.processing.validation import validate_inputs
from GradientBoostingClassifier.processing.data_management import load_pipeline
from GradientBoostingClassifier.config import config
from GradientBoostingClassifier import __version__ as _version


pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
_price_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(*, input_data) -> dict:
    """Make a prediction using the saved model pipeline."""

    data = pd.read_json(input_data)

    #validated_data = validate_inputs(input_data=data)


    output = _price_pipe.predict(data[config.features])
    #output = np.exp(prediction)

    results = {'predictions': output, 'version': _version}

    return results