from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from tqdm import tqdm


@dataclass
class Univariate:
    time_col: str
    target_col: str
    regressor_cols: Optional[List[str]]
    freq: str

    def get_indexed_series(self, data, col):
        ts = data.set_index(self.time_col).asfreq(self.freq)[col].copy()
        return ts

    def get_future_dataframe(self, data, fh=1):
        data_end = data[self.time_col].iloc[-1]
        future_start = data_end + pd.Timedelta(1, self.freq)
        future_range = pd.date_range(start=future_start, periods=fh, freq=self.freq, name=self.time_col)
        future_df = pd.DataFrame(future_range)
        return future_df

    def get_cv_slices(self, data, fh=1, start=0.5, stride=1):
        cv_slices = []
        n_obs = len(data)
        start_cutoff = int(start * n_obs)
        end_cutoff = n_obs - fh
        cutoffs = range(start_cutoff, end_cutoff+1, stride)
        for cutoff in cutoffs:
            data_slice = data.iloc[:cutoff, :]
            future_slice = self.get_future_dataframe(data_slice, fh=fh)
            if self.regressor_cols is not None:
                future_slice = future_slice.merge(data[[self.time_col] + self.regressor_cols], on=self.time_col)
            cv_slices.append((data_slice, future_slice))
        return cv_slices

    def historical_forecasts(self, data, fh=1, start=0.5, stride=1, progress=True):
        hfcst = []
        cv_slices = self.get_cv_slices(data, fh, start, stride)
        if progress:
            slices = tqdm(cv_slices)
        for train, future in slices:
            self.fit(train)
            fcst = self.predict(future)
            hfcst.append(fcst)
        return hfcst

    def plot_fcst(self, fcst, style="-"):
        target_fcst = self.target_col + "_fcst"
        target_lower = self.target_col + "_lower"
        target_upper = self.target_col + "_upper"
        f, ax = plt.subplots()
        y = self.get_indexed_series(self.data, self.target_col)
        y_fcst = self.get_indexed_series(fcst, target_fcst)
        y.plot(ax=ax, style=style)
        y_fcst.plot(ax=ax, style=style)
        if target_lower in fcst.columns and target_upper in fcst.columns:
            y_lower = self.get_indexed_series(fcst, target_lower)
            y_upper = self.get_indexed_series(fcst, target_upper)
            ax.fill_between(x=y_fcst.index, y1=y_lower, y2=y_upper, alpha=0.8, color="lightblue")
        plt.legend()

    def extend_fcst(self, data, fcst):
        fcst_ex = fcst.merge(data[[self.time_col, self.target_col]], on="date")
        return fcst_ex

    def error_table(self, data, hfcst):
        hfcst_ex = [self.extend_fcst(data, fcst) for fcst in hfcst]
        errors = [fcst[self.target_col] - fcst[self.target_col + "_fcst"] for fcst in hfcst_ex]
        err_table = pd.concat(errors, axis=1)
        return err_table

    def performance_metrics(self, data, hfcst, metrics=None, agg=None):
        pass



@dataclass
class Naive(Univariate):
    def fit(self, data):
        self.data = data.copy()
        self.last = data[self.target_col].iloc[-1]
        return self
        
    def predict(self, future):
        fh = len(future)
        future["y_fcst"] = np.array([self.last] * fh)
        return future


@dataclass
class AutoARIMA(Univariate):
    arima_params: dict = field(default_factory=dict)
    fit_params: dict = field(default_factory=dict)
    
    def fit(self, data):
        self.data = data.copy()
        self.model = pm.AutoARIMA(**self.arima_params)
        y = self.get_indexed_series(data, self.target_col)
        if self.regressor_cols is None:
            X = None
        else:
            X = self.get_indexed_series(data, self.regressor_cols)
        self.model.fit(y, X, **self.fit_params)
        return self
    
    def predict(self, future, conf_int=None):
        fh = len(future)
        if self.regressor_cols is None:
            X = None
        else:
            X = self.get_indexed_series(future, self.regressor_cols)
        if conf_int is None:
            y_fcst = self.model.predict(n_periods=fh, X=X)
            future[self.target_col + "_fcst"] = y_fcst
        else:
            y_fcst, y_conf = self.model.predict(n_periods=fh, X=X, return_conf_int=True, alpha=1-conf_int)
            future[self.target_col + "_fcst"] = y_fcst
            future[self.target_col + "_lower"] = y_conf[:, 0]
            future[self.target_col + "_upper"] = y_conf[:, 1]
        return future
