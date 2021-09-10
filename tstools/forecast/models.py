from dataclasses import dataclass, field
from typing import List, Optional, Union
from tstools.typing import ScikitModel, ScikitScaler
from tstools.forecast.base import Univariate, Multivariate

import numpy as np
import pandas as pd

# statsmodels 0.12.2
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.vector_ar.var_model import VAR as VARModel

# pmdarima 1.8.2
import pmdarima as pm


@dataclass
class AutoRegression(Univariate):
    lags: Union[int, List[int]] = 1
    trend: str = 'c'
    seasonal: bool = False
    period: Optional[int] = None
    autoreg_params: dict = field(default_factory=dict)
    fit_params: dict = field(default_factory=dict)

    def fit(self, data):
        self.prefit(data)
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
        self.prefit(data)
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
class AutoARIMA(Univariate):
    arima_params: dict = field(default_factory=dict)
    fit_params: dict = field(default_factory=dict)

    def fit(self, data):
        self.prefit(data)
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
        self.prefit(data)
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
            freq=self.freq,
            order=(0, 1, 0),
        )

    def fit(self, data):
        self.model.fit(data)
        self.data = self.model.data
        self.freq = self.model.freq
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
            freq=self.freq,
            order=(0, 1, 0),
            trend="t",
        )

    def fit(self, data):
        self.model.fit(data)
        self.data = self.model.data
        self.freq = self.model.freq
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
            freq=self.freq,
            order=(0, 0, 0),
            trend="c",
        )

    def fit(self, data):
        self.model.fit(data)
        self.data = self.model.data
        self.freq = self.model.freq
        return self

    def predict(self, future, conf_int=None):
        future = self.model.predict(future, conf_int)
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
        self.prefit(data)
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


@dataclass
class VAR(Multivariate):
    maxlags: Optional[int] = None
    trend: str = "c"
    var_params: dict = field(default_factory=dict)
    fit_params: dict = field(default_factory=dict)

    def fit(self, data):
        self.prefit(data)
        y = self.get_indexed_series(data, self.target_col)
        X = None
        if self.regressor_cols is not None:
            X = self.get_indexed_series(data, self.regressor_cols)
        self.model = VARModel(
            endog=y,
            exog=X,
            **self.var_params,
        )
        self.model_fit = self.model.fit(
            maxlags=self.maxlags,
            trend=self.trend,
            **self.fit_params,
        )
        return self

    def predict(self, future, conf_int=None):
        future = future.copy()
        alpha = 0.05
        if conf_int is not None:
            alpha = 1 - conf_int
        fh = len(future)
        y = self.get_indexed_series(self.data, self.target_col)
        X = None
        if self.regressor_cols is not None:
            X = self.get_indexed_series(future, self.regressor_cols).to_numpy()
        fcst = self.model_fit.forecast_interval(y.to_numpy(), exog_future=X, steps=fh, alpha=alpha)
        for i, target in enumerate(self.target_col):
            future[target + "_fcst"] = fcst[0][:, i]
            if conf_int is not None:
                future[target + "_lower"] = fcst[1][:, i]
                future[target + "_upper"] = fcst[2][:, i]
        return future
