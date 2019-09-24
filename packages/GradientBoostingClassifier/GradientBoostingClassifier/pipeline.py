from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from GradientBoostingClassifier.processing import preprocessors as pp
#from GradientBoostingClassifier.processing import features
from GradientBoostingClassifier.config import config



#_logger = logging.getLogger(__name__)


celcius = Pipeline(
    [
        ('LogTransform',
            pp.LogTransform(variables=config.log_transform))])