from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from tstools.metrics import mae as metric_mae


@dataclass
class BaseSingleSeries:
    time_col: str
    freq: Optional[str] = None
    regressor_cols: Optional[List[str]] = None

    def infer_frequency(self, data):
        freq = pd.infer_freq(data[self.time_col])
        return freq

    def get_indexed_series(self, data, col):
        ts = data.set_index(self.time_col).asfreq(self.freq)[col].copy()
        return ts

    def get_future_dataframe(self, data, fh=1):
        data_end = data[self.time_col].iloc[-1]
        future_start = data_end + pd.tseries.frequencies.to_offset(self.freq)  # Beware of this.
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
        self.prefit(data)
        hfcst = []
        cv_slices = self.get_cv_slices(data, fh, start, stride)
        if progress:
            slices = tqdm(cv_slices)
        for train, future in slices:
            self.fit(train)
            fcst = self.predict(future)
            hfcst.append(fcst)
        return hfcst

    def prefit(self, data):
        self.data = data.copy()
        if self.freq is None:
            self.freq = self.infer_frequency(data)


@dataclass
class Univariate(BaseSingleSeries):
    target_col: Optional[str] = None

    def __post_init__(self):
        if self.target_col is None:
            raise Exception("A target column must be provided.")
        if not isinstance(self.time_col, str):
            raise Exception("time_col should be a string")
        if not isinstance(self.target_col, str):
            raise Exception("target_col should be a string")
        if not isinstance(self.regressor_cols, list) and self.regressor_cols is not None:
            raise Exception("regressors_cols should be a List or None")

    def plot_fcst(self, fcst, train=None, test=None, plot_history=True, style="-", figsize=None):
        f, ax = plt.subplots(figsize=figsize)
        target_fcst = self.target_col + "_fcst"
        target_lower = self.target_col + "_lower"
        target_upper = self.target_col + "_upper"
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

    def score(self, fcst, val, metrics=None):
        fcst_ex = fcst.merge(val[[self.time_col, self.target_col]], on="date")
        y = self.get_indexed_series(fcst_ex, self.target_col)
        y_fcst = self.get_indexed_series(fcst_ex, self.target_col + "_fcst")
        scores_dict = dict()
        for metric in metrics:
            scores_dict[metric.__name__] = metric(y, y_fcst)
        scores_df = pd.Series(scores_dict)
        return scores_df

    def score_cv(self, hfcst, val, metrics=None, agg=None):
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
class Multivariate(BaseSingleSeries):
    target_col: List[str] = None

    def __post_init__(self):
        if self.target_col is None:
            raise Exception("A target column must be provided.")
        if not isinstance(self.time_col, str):
            raise Exception("time_col should be a string")
        if not isinstance(self.target_col, list):
            raise Exception("target_col should be a list")
        if not isinstance(self.regressor_cols, list) and self.regressor_cols is not None:
            raise Exception("regressors_cols should be a List or None")

    def plot_fcst(self, fcst, train=None, test=None, plot_history=True, style="-", figsize=None):
        f, axs = plt.subplots(nrows=len(self.target_col), figsize=figsize)
        if len(self.target_col) == 1:
            axs = [axs]
        for target, ax in zip(self.target_col, axs):
            target_fcst = target + "_fcst"
            target_lower = target + "_lower"
            target_upper = target + "_upper"

            if plot_history:
                if train is not None:
                    y = self.get_indexed_series(train, target)
                elif hasattr(self, "data"):
                    y = self.get_indexed_series(self.data, target)
                else:
                    raise Exception(
                        "Model has no historic data to plot. "
                        "Try with plot_history=False, "
                        "or use 'train' argument to supply train data."
                    )
                y.plot(ax=ax, style=style, color="black", label=target + "_train")
            if test is not None:
                y_test = self.get_indexed_series(test, target)
                y_test.plot(ax=ax, style=style, color="orange", label=target + "_test")
            y_fcst = self.get_indexed_series(fcst, target_fcst)
            y_fcst.plot(ax=ax, style=style, color="blue")
            if target_lower in fcst.columns and target_upper in fcst.columns:
                y_lower = self.get_indexed_series(fcst, target_lower)
                y_upper = self.get_indexed_series(fcst, target_upper)
                ax.fill_between(x=y_fcst.index, y1=y_lower, y2=y_upper, alpha=0.8, color="lightblue")
            ax.legend()
        plt.tight_layout()

    def score(self, fcst, val, metrics=None):
        fcst_ex = fcst.merge(val[[self.time_col] + self.target_col], on="date")
        target_scores = dict()
        for target in self.target_col:
            y = self.get_indexed_series(fcst_ex, target)
            y_fcst = self.get_indexed_series(fcst_ex, target + "_fcst")
            scores_dict = dict()
            for metric in metrics:
                scores_dict[metric.__name__] = metric(y, y_fcst)
            target_scores[target] = pd.Series(scores_dict)
        scores_df = pd.concat(target_scores, axis=1)
        return scores_df

    def score_cv(self, hfcst, val, metrics=None, agg=None):
        if metrics is None:
            metrics = [metric_mae]
        hfcst_ex = [fcst.merge(val[[self.time_col] + self.target_col], on="date") for fcst in hfcst]
        target_scores = dict()
        for target in self.target_col:
            scores_list = []
            for fcst in hfcst_ex:
                y = self.get_indexed_series(fcst, target)
                y_fcst = self.get_indexed_series(fcst, target + "_fcst")
                step_scores = dict()
                for metric in metrics:
                    step_scores[metric.__name__] = metric(y, y_fcst)
                scores_list.append(pd.Series(step_scores))
            target_scores[target] = pd.DataFrame(scores_list)
        scores_df = pd.concat(target_scores, axis=1)
        if agg is not None:
            return scores_df.apply(agg, axis=0).unstack(level=0)
        return scores_df
