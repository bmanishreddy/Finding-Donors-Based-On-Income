import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler



#Log transform the data

#One hot encode the data

class LogTransform(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None) -> None:

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):
        #print(variables)


        return self
    def transform(self,X):
        X = X.copy()
        X[self.variables] = X[self.variables].apply(lambda x: np.log(x + 1))


        return X

class MinMaxScalerTransform(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None) -> None:

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables


    def fit(self,X,y=None):

        self.scalar = MinMaxScaler()

        return self

    def transform(self,X):
        X = X.copy()

        X[self.variables] = self.scalar.fit_transform(X[self.variables])

        return X


#Encoding


'''
class OneHotEncodingFeatures(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None) -> None:

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):
        return self

    def transform(self,X):



        X = X.copy()

        X = pd.get_dummies(X[self.variables])

        print(X)


        return X
        
        
'''

#Target into train test splits

'''
class TargetEncoding(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None) -> None:

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        print("log one ---")

    def fit(self,y,X=None):
        return self

    def transform(self,y):



        y = y.copy()

        y = y.replace({'<=50K':0, '>50K':1})


        print("inside transform ==", y)

        return y'''
