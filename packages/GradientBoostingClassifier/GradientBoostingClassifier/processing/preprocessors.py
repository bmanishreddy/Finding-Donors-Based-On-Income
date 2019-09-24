import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


#Log transform the data

#One hot encode the data

class LogTransform(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
    def fit(self,X,y=None):
        
        return
    def transform(self):
        return