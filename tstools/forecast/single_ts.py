from dataclasses import dataclass, field
from typing import List, Optional, Union
from tstools.typing import ScikitModel, ScikitScaler
from tstools.metrics import mae as metric_mae

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# statsmodels 0.12.2
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# pmdarima 1.8.2
import pmdarima as pm


@dataclass
class SingleTS:
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


@dataclass
class Univariate(SingleTS):
    time_col: str
    target_col: str
    regressor_cols: Optional[List[str]]
    freq: str

    def __post_init__(self):
        if not isinstance(self.time_col, str):
            raise Exception("time_col should be a string")
        if not isinstance(self.target_col, str):
            raise Exception("target_col should be a string")
        if not isinstance(self.regressor_cols, list) and self.regressor_cols is not None:
            raise Exception("regressors_cols should be a List or None")
        if not isinstance(self.freq, str):
            raise Exception("freq should be a string")

    def plot_fcst(self, fcst, train=None, test=None, plot_history=True, style="-"):
        target_fcst = self.target_col + "_fcst"
        target_lower = self.target_col + "_lower"
        target_upper = self.target_col + "_upper"
        f, ax = plt.subplots()
        if plot_history:
            if train is not None:
                y = self.get_indexed_series(train, self.target_col)
            elif hasattr(self, "data"):
                y = self.get_indexed_series(self.data, self.target_col)
            else:
                raise Exception(
                    "Model has no historic data to plot. "
                    "Try with plot_history=False, "
                    "or use 'train' argument to supply train data."
                )
            y.plot(ax=ax, style=style, color="black", label=self.target_col + "_train")
        if test is not None:
            y_test = self.get_indexed_series(test, self.target_col)
            y_test.plot(ax=ax, style=style, color="orange", label=self.target_col + "_test")
        y_fcst = self.get_indexed_series(fcst, target_fcst)
        y_fcst.plot(ax=ax, style=style, color="blue")
        if target_lower in fcst.columns and target_upper in fcst.columns:
            y_lower = self.get_indexed_series(fcst, target_lower)
            y_upper = self.get_indexed_series(fcst, target_upper)
            ax.fill_between(x=y_fcst.index, y1=y_lower, y2=y_upper, alpha=0.8, color="lightblue")
        plt.legend()

    def score(self, val, fcst, metrics=None):
        fcst_ex = fcst.merge(val[[self.time_col, self.target_col]], on="date")
        y = self.get_indexed_series(fcst_ex, self.target_col)
        y_fcst = self.get_indexed_series(fcst_ex, self.target_col + "_fcst")
        scores_dict = dict()
        for metric in metrics:
            scores_dict[metric.__name__] = metric(y, y_fcst)
        scores_df = pd.DataFrame(scores_dict, index=[0])
        return scores_df

    def score_cv(self, val, hfcst, metrics=None, agg=None):
        if metrics is None:
            metrics = [metric_mae]
        hfcst_ex = [fcst.merge(val[[self.time_col, self.target_col]], on="date") for fcst in hfcst]
        scores_list = []
        for fcst in hfcst_ex:
            y = self.get_indexed_series(fcst, self.target_col)
            y_fcst = self.get_indexed_series(fcst, self.target_col + "_fcst")
            step_scores = dict()
            for metric in metrics:
                step_scores[metric.__name__] = metric(y, y_fcst)
            scores_list.append(pd.Series(step_scores))
        scores_df = pd.DataFrame(scores_list)
        if agg is not None:
            return scores_df.apply(agg, axis=0)
        return scores_df


@dataclass
class Multivariate(SingleTS):
    time_col: str
    target_col: List[str]
    regressor_cols: Optional[List[str]]
    freq: str

    def __post_init__(self):
        if not isinstance(self.time_col, str):
            raise Exception("time_col should be a string")
        if not isinstance(self.target_col, list):
            raise Exception("target_col should be a list")
        if not isinstance(self.regressor_cols, list) and self.regressor_cols is not None:
            raise Exception("regressors_cols should be a List or None")
        if not isinstance(self.freq, str):
            raise Exception("freq should be a string")


@dataclass
class AutoRegression(Univariate):
    lags: Union[int, List[int]] = 1
    trend: str = 'c'
    seasonal: bool = False
    period: Optional[int] = None
    autoreg_params: dict = field(default_factory=dict)
    fit_params: dict = field(default_factory=dict)

    def fit(self, data):
        self.data = data.copy()
        y = self.get_indexed_series(data, self.target_col)
        X = None
        if self.regressor_cols is not None:
            X = self.get_indexed_series(data, self.regressor_cols)
        self.model = AutoReg(
            endog=y,
            exog=X,
            lags=self.lags,
            trend=self.trend,
            period=self.period,
            **self.autoreg_params,
        )
        self.model_fit = self.model.fit(**self.fit_params)
        return self

    def predict(self, future, conf_int=None):
        future = future.copy()
        alpha = 0.05
        if conf_int is not None:
            alpha = 1 - conf_int
        X = None
        if self.regressor_cols is not None:
            X = self.get_indexed_series(future, self.regressor_cols)
        fcst = fcst = self.model_fit.get_prediction(
            start=future[self.time_col].iloc[0],
            end=future[self.time_col].iloc[-1],
            exog_oos=X,
        ).summary_frame(alpha=alpha)
        future[self.target_col + "_fcst"] = fcst["mean"].to_numpy()
        if conf_int is not None:
            future[self.target_col + "_lower"] = fcst["mean_ci_lower"].to_numpy()
            future[self.target_col + "_upper"] = fcst["mean_ci_upper"].to_numpy()
        return future


@dataclass
class ARIMA(Univariate):
    order: tuple = (0, 0, 0)
    seasonal_order: tuple = (0, 0, 0, 0)
    trend: Optional[str] = None
    arima_params: dict = field(default_factory=dict)
    fit_params: dict = field(default_factory=dict)

    def fit(self, data):
        self.data = data.copy()
        y = self.get_indexed_series(data, self.target_col)
        X = None
        if self.regressor_cols is not None:
            X = self.get_indexed_series(data, self.regressor_cols)
        self.model = ARIMAModel(
            endog=y,
            exog=X,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            **self.arima_params,
        )
        self.model_fit = self.model.fit(**self.fit_params)
        return self

    def predict(self, future, conf_int=None):
        future = future.copy()
        alpha = 0.05
        if conf_int is not None:
            alpha = 1 - conf_int
        fh = len(future)
        X = None
        if self.regressor_cols is not None:
            X = self.get_indexed_series(future, self.regressor_cols)
        fcst = self.model_fit.get_forecast(steps=fh, exog=X).summary_frame(alpha=alpha)
        future[self.target_col + "_fcst"] = fcst["mean"].to_numpy()
        if conf_int is not None:
            future[self.target_col + "_lower"] = fcst["mean_ci_lower"].to_numpy()
            future[self.target_col + "_upper"] = fcst["mean_ci_upper"].to_numpy()
        return future


@dataclass
class ETS(Univariate):
    error: str = 'add'
    trend: str = None
    damped_trend: bool = False
    seasonal: str = None
    seasonal_periods: int = None
    ets_params: dict = field(default_factory=dict)
    fit_params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.regressor_cols is not None:
            raise Exception("ETS model does not accept regressors")

    def fit(self, data):
        self.data = data.copy()
        y = self.get_indexed_series(data, self.target_col)
        self.model = ETSModel(
            endog=y,
            error=self.error,
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            **self.ets_params,
        )
        self.model_fit = self.model.fit(**self.fit_params)
        return self

    def predict(self, future, conf_int=None):
        future = future.copy()
        alpha = 0.05
        if conf_int is not None:
            alpha = 1 - conf_int
        fcst = self.model_fit.get_prediction(
            start=future[self.time_col].iloc[0], end=future[self.time_col].iloc[-1]).summary_frame(alpha=alpha)
        future[self.target_col + "_fcst"] = fcst["mean"].to_numpy()
        if conf_int is not None:
            future[self.target_col + "_lower"] = fcst["pi_lower"].to_numpy()
            future[self.target_col + "_upper"] = fcst["pi_upper"].to_numpy()
        return future


@dataclass
class Naive(Univariate):
    def __post_init__(self):
        if self.regressor_cols is not None:
            raise Exception("Naive model does not accept regressors")
        self.model = ARIMA(
            time_col=self.time_col,
            target_col=self.target_col,
            regressor_cols=None,
            freq="W",
            order=(0, 1, 0),
        )

    def fit(self, data):
        self.data = data.copy()
        self.model.fit(data)
        return self

    def predict(self, future, conf_int=None):
        future = self.model.predict(future, conf_int)
        return future


@dataclass
class Drift(Univariate):
    def __post_init__(self):
        if self.regressor_cols is not None:
            raise Exception("Drift model does not accept regressors")
        self.model = ARIMA(
            time_col=self.time_col,
            target_col=self.target_col,
            regressor_cols=None,
            freq="W",
            order=(0, 1, 0),
            trend="t",
        )

    def fit(self, data):
        self.data = data.copy()
        self.model.fit(data)
        return self

    def predict(self, future, conf_int=None):
        future = self.model.predict(future, conf_int)
        return future


@dataclass
class Mean(Univariate):
    def __post_init__(self):
        if self.regressor_cols is not None:
            raise Exception("Mean model does not accept regressors")
        self.model = ARIMA(
            time_col=self.time_col,
            target_col=self.target_col,
            regressor_cols=None,
            freq="W",
            order=(0, 0, 0),
            trend="c",
        )

    def fit(self, data):
        self.data = data.copy()
        self.model.fit(data)
        return self

    def predict(self, future, conf_int=None):
        future = self.model.predict(future, conf_int)
        return future


@dataclass
class AutoARIMA(Univariate):
    arima_params: dict = field(default_factory=dict)
    fit_params: dict = field(default_factory=dict)

    def fit(self, data):
        self.data = data.copy()
        self.model = pm.AutoARIMA(**self.arima_params)
        y = self.get_indexed_series(data, self.target_col)
        X = None
        if self.regressor_cols is not None:
            X = self.get_indexed_series(data, self.regressor_cols)
        self.model.fit(y, X, **self.fit_params)
        return self

    def predict(self, future, conf_int=None):
        future = future.copy()
        fh = len(future)
        X = None
        if self.regressor_cols is not None:
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


@dataclass
class ScikitRegression(Univariate):
    model: Optional[ScikitModel] = None
    scaler: Optional[ScikitScaler] = None
    n_lags: Optional[int] = None

    def build_lags(self, ts):
        lags = pd.concat([ts.shift(i) for i in range(self.n_lags)], axis=1).dropna()
        lags.columns = [ts.name + f"_lag_{i}" for i in range(self.n_lags)]
        return lags

    def build_target(self, ts):
        target = ts.shift(-1).dropna()
        target.name = "target"
        return target

    def fit(self, data):
        self.data = data.copy()
        ts = self.get_indexed_series(data, self.target_col)
        lags = None
        extra_regressors = None
        if self.n_lags is not None:
            lags = self.build_lags(ts)
        if self.regressor_cols is not None:
            extra_regressors = self.get_indexed_series(data, self.regressor_cols)
        target = self.build_target(ts)
        df = pd.concat([target, lags, extra_regressors], axis=1, join="inner")
        y = df.iloc[:, 0].to_numpy()
        X = df.iloc[:, 1:].to_numpy()
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self.X_fit = X.copy()
        self.y_fit = y.copy()
        return self

    def predict(self, future):
        future = future.copy()
        fh = len(future)
        ts = self.get_indexed_series(self.data, self.target_col).to_numpy()
        extra_regressors = None
        if self.regressor_cols is not None:
            extra_regressors = self.get_indexed_series(future, self.regressor_cols).to_numpy()
        for i, row in future.iterrows():
            X_components = []
            if self.n_lags is not None:
                lags = np.flip(ts[-self.n_lags:].reshape(1, -1))
                X_components.append(lags)
            if extra_regressors is not None:
                reg = extra_regressors[i].reshape(1, -1)
                X_components.append(reg)
            X = np.concatenate(X_components, axis=1)
            if self.scaler is not None:
                X = self.scaler.transform(X)
            y_pred = self.model.predict(X)
            ts = np.append(ts, [y_pred])
        future[self.target_col + "_fcst"] = ts[-fh:]
        return future

    def residuals(self):
        resid = self.y_fit - self.model.predict(self.X_fit)
        return resid
