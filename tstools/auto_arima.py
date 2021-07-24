from dataclasses import dataclass, field

import pandas as pd
import pmdarima as pm
from src.ts_tools import BaseForecaster
from tqdm import tqdm


@dataclass
class AutoARIMA(BaseForecaster):
    arima_params: dict = field(default_factory=dict)
    fit_params: dict = field(default_factory=dict)

    def fit_for_each_key(self, data, progress):
        model_dict = dict()
        data_keys = self.get_keys(data)
        if progress:
            data_keys = tqdm(data_keys)
        for key in data_keys:
            model_dict[key] = self.fit_for_each_target(data, key)
        return model_dict

    def fit_for_each_target(self, data, key):
        model_dict = dict()
        for target in self.target_cols:
            y = self.get_indexed_series(data, key, target)
            if self.predictor_cols is None:
                X = None
            else:
                X = self.get_indexed_series(data, key, self.predictor_cols)
            model = pm.AutoARIMA(**self.arima_params)
            model.fit(y, X, **self.fit_params)
            model_dict[target] = model
        return model_dict

    def predict_for_each_key(self, fh, predictors, conf_int, progress):
        fcst_dict = dict()
        model_items = self.model_dict.items()
        if progress:
            model_items = tqdm(model_items)
        for key, models_for_key in model_items:
            fcst_dict[key] = self.predict_for_each_target(key, models_for_key, fh, predictors, conf_int)
        fcst = pd.concat(fcst_dict.values())
        return fcst

    def predict_for_each_target(self, key, models_for_key, fh, predictors, conf_int):
        fcst_dict = dict()
        for target, model in models_for_key.items():
            y_fcst_idx = self.build_fcst_index(self.data, fh)
            if self.predictor_cols is None:
                X = None
            else:
                X = self.get_indexed_series(predictors, key, self.predictor_cols)
            if conf_int is None:
                y_fcst = model.predict(n_periods=fh, X=X)
                y_fcst = pd.DataFrame(y_fcst, index=y_fcst_idx, columns=[target + "_fcst"])
            else:
                y_fcst, y_conf = model.predict(n_periods=fh, X=X, return_conf_int=True, alpha=1-conf_int)
                y_fcst = pd.DataFrame(y_fcst, index=y_fcst_idx, columns=[target + "_fcst"])
                y_fcst[target + "_lower"] = y_conf[:, 0]
                y_fcst[target + "_upper"] = y_conf[:, 1]
            y_fcst = y_fcst.reset_index()
            fcst_dict[target] = y_fcst
        fcst_df = pd.concat({target: fcst.set_index(self.time_col) for target, fcst in fcst_dict.items()}, axis=1)
        fcst_df.columns = fcst_df.columns.droplevel()
        fcst_df = fcst_df.reset_index()
        fcst_df.insert(0, self.key_col, key)
        return fcst_df

    def fit(self, data, progress=True):
        self.data = data.copy()
        self.model_dict = self.fit_for_each_key(data, progress)
        return self

    def predict(self, fh=1, predictors=None, conf_int=None, progress=True):
        fcst = self.predict_for_each_key(fh, predictors, conf_int, progress)
        return fcst
