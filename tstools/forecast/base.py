from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from tstools.metrics import mae as metric_mae


@dataclass
class BaseSingleTS:
    time_col: str
    freq: Optional[str] = None
    target_cols: Optional[List[str]] = None
    regressor_cols: Optional[List[str]] = None

    def __post_init__(self):
        if not isinstance(self.time_col, str):
            raise Exception("time_col should be a string")
        if self.target_cols is None:
            raise Exception("A target column must be provided.")
        if not isinstance(self.target_cols, list):
            raise Exception("target_col should be a list")
        if not isinstance(self.regressor_cols, list) and self.regressor_cols is not None:
            raise Exception("regressors_cols should be a List or None")

    def infer_frequency(self, data):
        freq = pd.infer_freq(data[self.time_col])
        return freq

    def prefit(self, data):
        self.data = data.copy()
        if self.freq is None:
            self.freq = self.infer_frequency(data)

    def get_indexed_series(self, data, col):
        ts = data.set_index(self.time_col).asfreq(self.freq)[col].copy()
        return ts

    def get_future_dataframe(self, data, fh=1):
        data_end = data[self.time_col].iloc[-1]
        # Beware of the following line. The function 'to_offset' may change in the future.
        future_start = data_end + pd.tseries.frequencies.to_offset(self.freq)
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

    def plot_fcst(self, fcst, train=None, test=None, plot_history=True, style="-", figsize=None):
        f, axs = plt.subplots(nrows=len(self.target_cols), figsize=figsize)
        if len(self.target_cols) == 1:
            axs = [axs]
        for target, ax in zip(self.target_cols, axs):
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
        fcst_ex = fcst.merge(val[[self.time_col] + self.target_cols], on="date")
        target_scores = dict()
        for target in self.target_cols:
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
        hfcst_ex = [fcst.merge(val[[self.time_col] + self.target_cols], on="date") for fcst in hfcst]
        target_scores = dict()
        for target in self.target_cols:
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


@dataclass
class BaseUnivariate(BaseSingleTS):
    def __post_init__(self):
        super().__post_init__()
        if not len(self.target_cols) == 1:
            raise Exception("A list with a single target should be provided for univariate models")
        self.target = self.target_cols[0]


@dataclass
class BaseMultiTS:
    key_col: str
    time_col: str
    freq: Optional[str] = None
    target_cols: Optional[List[str]] = None
    regressor_cols: Optional[List[str]] = None

    def __post_init__(self):
        if not isinstance(self.key_col, str):
            raise Exception("key_col should be a string")
        if not isinstance(self.time_col, str):
            raise Exception("time_col should be a string")
        if self.target_cols is None:
            raise Exception("A target column must be provided.")
        if not isinstance(self.target_cols, list):
            raise Exception("target_col should be a list")
        if not isinstance(self.regressor_cols, list) and self.regressor_cols is not None:
            raise Exception("regressors_cols should be a List or None")

    def get_keys(self, data):
        return list(data[self.key_col].unique())

    def get_ts(self, data, key):
        ts = data[data[self.key_col] == key].copy()
        return ts

    def get_indexed_series(self, data, key, col):
        ts = self.get_ts(data, key)
        y = ts.set_index(self.time_col).asfreq(self.freq)[col].copy()
        return y

    def infer_frequency(self, data):
        keys = self.get_keys(data)
        freq_list = []
        for key in keys:
            ts = self.get_ts(data, key)
            ts_freq = pd.infer_freq(ts[self.time_col])
            freq_list.append(ts_freq)
        all_freqs_are_equal = all(f == freq_list[0] for f in freq_list)
        if not all_freqs_are_equal:
            print("Warning: frequency could not be inferred because not all the series have the same frequency")
            return None
        return freq_list[0]

    def prefit(self, data):
        self.data = data.copy()
        if self.freq is None:
            self.freq = self.infer_frequency(data)

    def get_future_dataframe(self, data, fh=1):
        future_list = []
        for key in self.get_keys(data):
            ts = self.get_ts(data, key)
            ts_end = ts[self.time_col].iloc[-1]
            # Beware of the following line. The function 'to_offset' may change in the future.
            future_start = ts_end + pd.tseries.frequencies.to_offset(self.freq)
            future_range = pd.date_range(start=future_start, periods=fh, freq=self.freq, name=self.time_col)
            future_df = pd.DataFrame({
                self.key_col: (key for _ in range(len(future_range))),
                self.time_col: future_range,
            })
            future_list.append(future_df)
        future_by_key = pd.concat(future_list).reset_index(drop=True)
        return future_by_key

    def get_cv_slices(self, data, fh=1, start=0.5, stride=1):
        cv_slices = []
        data_by_key = data.groupby(self.key_col, group_keys=False)
        n_obs = int(len(data) / len(data_by_key))
        start_cutoff = int(start * n_obs)
        end_cutoff = n_obs - fh
        cutoffs = range(start_cutoff, end_cutoff+1, stride)
        for cutoff in cutoffs:
            print(cutoff)
            data_slice = data_by_key.apply(lambda df: df.iloc[:cutoff, :])
            future_slice = self.get_future_dataframe(data_slice, fh=fh)
            if self.regressor_cols is not None:
                future_slice = future_slice.merge(
                    data[[self.key_col, self.time_col] + self.regressor_cols],
                    on=[self.key_col, self.time_col],
                )
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

    def plot_fcst(self, key, fcst, train=None, test=None, plot_history=True, style="-", figsize=None):
        f, axs = plt.subplots(nrows=len(self.target_cols), figsize=figsize)
        if len(self.target_cols) == 1:
            axs = [axs]
        for target, ax in zip(self.target_cols, axs):
            target_fcst = target + "_fcst"
            target_lower = target + "_lower"
            target_upper = target + "_upper"
            if plot_history:
                if train is not None:
                    y = self.get_indexed_series(train, key, target)
                elif hasattr(self, "data"):
                    y = self.get_indexed_series(self.data, key, target)
                else:
                    raise Exception(
                        "Model has no historic data to plot. "
                        "Try with plot_history=False, "
                        "or use 'train' argument to supply train data."
                    )
                y.plot(ax=ax, style=style, color="black", label=target + "_train")
            if test is not None:
                y_test = self.get_indexed_series(test, key, target)
                y_test.plot(ax=ax, style=style, color="orange", label=target + "_test")
            y_fcst = self.get_indexed_series(fcst, key, target_fcst)
            y_fcst.plot(ax=ax, style=style, color="blue")
            if target_lower in fcst.columns and target_upper in fcst.columns:
                y_lower = self.get_indexed_series(fcst, key, target_lower)
                y_upper = self.get_indexed_series(fcst, key, target_upper)
                ax.fill_between(x=y_fcst.index, y1=y_lower, y2=y_upper, alpha=0.8, color="lightblue")
            ax.legend()
        plt.tight_layout()

    def score(self, fcst, val, metrics=None):
        fcst_ex = fcst.merge(
            val[[self.key_col, self.time_col] + self.target_cols],
            on=[self.key_col, self.time_col],
        )
        key_scores = dict()
        for key in self.get_keys(val):
            target_scores = dict()
            for target in self.target_cols:
                y = self.get_indexed_series(fcst_ex, key, target)
                y_fcst = self.get_indexed_series(fcst_ex, key, target + "_fcst")
                scores_dict = dict()
                for metric in metrics:
                    scores_dict[metric.__name__] = metric(y, y_fcst)
                target_scores[target] = pd.Series(scores_dict)
            scores_df = pd.concat(target_scores, axis=1)
            key_scores[key] = scores_df
        return key_scores

    def score_cv(self, hfcst, val, metrics=None, agg=None):
        if metrics is None:
            metrics = [metric_mae]
        hfcst_ex = [fcst.merge(
            val[[self.key_col, self.time_col] + self.target_cols],
            on=[self.key_col, self.time_col]) for fcst in hfcst]
        key_scores = dict()
        for key in self.get_keys(val):
            target_scores = dict()
            for target in self.target_cols:
                scores_list = []
                for fcst in hfcst_ex:
                    y = self.get_indexed_series(fcst, key, target)
                    y_fcst = self.get_indexed_series(fcst, key, target + "_fcst")
                    step_scores = dict()
                    for metric in metrics:
                        step_scores[metric.__name__] = metric(y, y_fcst)
                    scores_list.append(pd.Series(step_scores))
                target_scores[target] = pd.DataFrame(scores_list)
            scores_df = pd.concat(target_scores, axis=1)
            key_scores[key] = scores_df
        # if agg is not None:
        #     return scores_df.apply(agg, axis=0).unstack(level=0)
        return key_scores
