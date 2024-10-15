from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

class XGBoost:
    def __init__(self, random_seed=42, **params):
        self.params = params
        self.random_seed = random_seed
    
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        self.model = XGBRegressor(**self.params, method="hist", device="cuda", random_state=self.random_seed)
        if X_valid == None and y_valid == None:
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    
    def predict(self, X_valid):
        if self.model == None:
            raise ValueError("model is not trained.")
        return self.model.predict(X_valid)


class LightGBM:
    def __init__(self, random_seed=42, **params):
        self.params = params
        self.random_seed = random_seed
    
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        self.model = LGBMRegressor(**self.params, method="hist", device="cuda", random_state=self.random_seed)
        if X_valid == None and y_valid == None:
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    
    def predict(self, X_valid):
        if self.model == None:
            raise ValueError("model is not trained.")
        return self.model.predict(X_valid)


class CatBoost:
    def __init__(self, random_seed=42, **params):
        self.params = params
        self.random_seed = random_seed
    
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        self.model = CatBoostRegressor(**self.params, method="hist", device="cuda", random_state=self.random_seed)
        if X_valid == None and y_valid == None:
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    
    def predict(self, X_valid):
        if self.model == None:
            raise ValueError("model is not trained.")
        return self.model.predict(X_valid)