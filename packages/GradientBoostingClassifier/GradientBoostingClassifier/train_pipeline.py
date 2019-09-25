import numpy as np
from sklearn.model_selection import train_test_split

from GradientBoostingClassifier import pipeline
from GradientBoostingClassifier.processing.data_management import (
    load_dataset, save_pipeline)
from GradientBoostingClassifier.config import config
#from GradientBoostingClassifier import __version__ as _version

#import logging


#_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.features],
        data[config.target],
        test_size=0.1,
        random_state=0)  # we are setting the seed here

    y_train = y_train.replace({'<=50K': 0, '>50K': 1})
    y_test = y_test.replace({'<=50K': 0, '>50K': 1})

    # transform the target
    #print(y_train)

    pipeline.celcius.fit(X_train[config.features],
                            y_train)

    #_logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.celcius)


if __name__ == '__main__':
    run_training()
