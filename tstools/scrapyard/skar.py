from dataclasses import dataclass

import pandas as pd
from src.tstools import BaseForecaster
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from typing_extensions import Protocol
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X): ...
    def score(self, X, y, sample_weight=None): ...
    def set_params(self, **params): ...
    def get_params(self, deep=True): ...


@dataclass
class SKAR(BaseForecaster):
    model: ScikitModel = None
    scale_regressors: bool = True
    n_lags: int = 0

    def build_lags(self, y):
        lags = pd.concat([y.shift(i) for i in range(self.n_lags + 1)], axis=1).dropna()
        return lags

    def build_target(self, y):
        target = pd.concat([y.shift(-i) for i in range(1, self.fh + 1)], axis=1).dropna()
        return target

    def build_fcst_index(self):
        data_end = self.data[self.time_col].iloc[-1]
        fcst_start = data_end + pd.Timedelta(1, self.freq)
        fcst_index = pd.date_range(start=fcst_start, periods=self.fh, freq=self.freq)
        return fcst_index

    def fit(self, data):
        if self.model is None:
            self.model = LinearRegression()
        self.data = data.copy()
        self.model_dict = dict()
        self.scaler_dict = dict()
        for key in self.get_keys(data):
            model_subdict = dict()
            scaler_subdict = dict()
            for target in self.target_cols:
                target_data = self.get_indexed_series(data, key, target)
                if self.predictor_cols is not None:
                    predictor_data = self.get_indexed_series(data, key, self.predictor_cols)
                else:
                    predictor_data = None
                y_target = self.build_target(target_data)
                y_lags = self.build_lags(target_data)
                YX = pd.concat([y_target, y_lags, predictor_data], axis=1, join="inner").to_numpy()
                Y = YX[:, :self.fh]
                X = YX[:, self.fh:]
                if self.scale_regressors:
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                    scaler_subdict[target] = scaler
                model = deepcopy(self.model)
                model.fit(X, Y)
                model_subdict[target] = model
            self.model_dict[key] = model_subdict
            self.scaler_dict[key] = scaler_subdict
        return self

    def predict(self, fh=None, predictors=None):
        fcst_dict = dict()
        for key, model_subdict in self.model_dict.items():
            fcst_subdict = dict()
            for target, model in model_subdict.items():
                target_data = self.get_indexed_series(self.data, key, [target])
                y_lags = np.flip(target_data.tail(self.n_lags+1).to_numpy())
                if self.predictor_cols is not None:
                    predictor_data = self.get_indexed_series(self.data, key, self.predictor_cols)
                    last_predictors = predictor_data.tail(1).to_numpy()
                    X = np.concatenate([y_lags, last_predictors]).T
                else:
                    X = y_lags.T
                if self.scale_regressors:
                    scaler = self.scaler_dict[key][target]
                    X = scaler.transform(X)
                y_fcst = model.predict(X)
                y_fcst_index = self.build_fcst_index()
                y_fcst = pd.Series(y_fcst.flatten(), index=y_fcst_index)
                y_fcst.index.name = self.time_col
                y_fcst.name = target + "_fcst"
                y_fcst = y_fcst.reset_index()
                fcst_subdict[target] = y_fcst
            fcst_subdf = pd.concat({key: fcst.set_index("date") for key, fcst in fcst_subdict.items()}, axis=1)
            fcst_subdf.columns = fcst_subdf.columns.droplevel()
            fcst_subdf = fcst_subdf.reset_index()
            fcst_subdf.insert(0, self.key_col, key)
            fcst_dict[key] = fcst_subdf
        fcst = pd.concat(fcst_dict.values())
        return fcst
