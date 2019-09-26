from regression_model.config import config

import pandas as pd


def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for unprocessable values."""

    validated_data = input_data.copy()

    # check for numerical variables with NA not seen during training
    if input_data[config.numerical].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.numerical)

    # check for categorical variables with NA not seen during training
    if input_data[config.stringfeatures].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.stringfeatures)


    return validated_data
