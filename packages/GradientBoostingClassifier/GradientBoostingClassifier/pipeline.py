from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import MinMaxScaler

from GradientBoostingClassifier.processing import preprocessors as pp
#from GradientBoostingClassifier.processing import features
from GradientBoostingClassifier.config import config

from sklearn.ensemble import GradientBoostingClassifier


#_logger = logging.getLogger(__name__)



celcius = Pipeline(

    [('LogTransform',pp.LogTransform(variables=config.log_transform)),
     ('MinMaxScalerTransform',pp.MinMaxScalerTransform(variables=config.numerical)),


     ('OneHotEncodingFeatures',pp.OneHotEncodingFeatures(variables=config.stringfeatures)),


      ('GradientBoostingClassifier', GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.2, loss='deviance', max_depth=8,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=15,
              presort='auto', random_state=1, subsample=1.0, verbose=0,
              warm_start=False))

     ]


)
#print(config.numerical)


'''celcius = Pipeline(
    [
        ('LogTransform',
            pp.LogTransform(variables=config.log_transform)),('MinMaxScalerTransform',
            pp.MinMaxScalerTransform(variables=config.numerical))])'''