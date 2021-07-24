from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipywidgets import interact, widgets
from src.ts_tools.metrics import error, mae
from tqdm import tqdm


@dataclass
class BaseForecaster:
    key_col: str
    time_col: str
    target_cols: List[str]
    predictor_cols: Optional[List[str]]
    freq: str
    fh: Optional[int] = None

    def get_keys(self, data):
        return list(data[self.key_col].unique())

    def get_group(self, data, key):
        subdata = data[data[self.key_col] == key].copy()
        return subdata

    def get_indexed_series(self, data, key, col):
        subdata = self.get_group(data, key)
        y = subdata.set_index(self.time_col).asfreq(self.freq)[col].copy()
        return y

    def get_data_slices(self, data, fh=1, start=0.5, stride=1):
        train_list = []
        data_by_key = data.groupby(self.key_col, group_keys=False)
        n_obs = int(len(data) / len(data_by_key))
        start_cutoff = int(start * n_obs)
        end_cutoff = n_obs - fh
        cutoffs = range(start_cutoff, end_cutoff+1, stride)
        for cutoff in cutoffs:
            data_slice = data_by_key.apply(lambda df: df.iloc[:cutoff, :])
            train_list.append(data_slice)
        return train_list

    def get_predictor_slices(self, predictors, fh=1, start=0.5, stride=1):
        predictors = predictors.groupby(self.key_col, group_keys=False).apply(lambda df: df.shift(-fh))
        predictor_slices = []
        for pred_slice in self.get_data_slices(predictors, fh, start, stride):
            predictor_slices.append(pred_slice.groupby(self.key_col).apply(lambda df: df.tail(fh)))
        return predictor_slices

    def build_fcst_index(self, data, fh=1):
        if self.fh is not None:
            fh = self.fh
        data_end = data[self.time_col].iloc[-1]
        fcst_start = data_end + pd.Timedelta(1, self.freq)
        fcst_index = pd.date_range(start=fcst_start, periods=fh, freq=self.freq)
        fcst_index.name = self.time_col
        return fcst_index

    def af_table(self, y, y_fcst, how="inner"):
        af_df = pd.merge(y, y_fcst, left_index=True, right_index=True, how=how)
        af_df.columns = ["y", "y_fcst"]
        af_df["error"] = af_df["y"] - af_df["y_fcst"]
        return af_df

    def af_trim(self, y, y_fcst, how="inner"):
        af_df = self.af_table(y, y_fcst, how=how)
        return af_df["y"], af_df["y_fcst"]

    def historical_forecasts(self, data, predictors=None, fh=1, start=0.5, stride=1, progress=True):
        if self.fh is not None:
            fh = self.fh
        hfcst = []
        train_slices = self.get_data_slices(data, fh, start, stride)
        if self.predictor_cols is None or predictors is None:
            predictor_slices = len(train_slices) * [None]
        else:
            predictor_slices = self.get_predictor_slices(predictors, fh, start, stride)
        slices = zip(train_slices, predictor_slices)
        if progress:
            slices = tqdm(list(slices))
        for train, predictor in slices:
            self.fit(train, progress=False)
            fcst = self.predict(fh, predictor, progress=False)
            hfcst.append(fcst)
        return hfcst

    def error_table(self, data, predictors=None, fh=1, start=0.5, stride=1, progress=True):
        hfcst = self.historical_forecasts(data, predictors, fh=fh, start=start, stride=stride, progress=progress)
        metrics_dict = dict()
        for key in self.get_keys(data):
            target_metrics = dict()
            for target in self.target_cols:
                metrics_list = list()
                for i, fcst in enumerate(hfcst):
                    y = self.get_indexed_series(data, key, target)
                    y_fcst = self.get_indexed_series(fcst, key, target + "_fcst")
                    y, y_fcst = self.af_trim(y, y_fcst)
                    err = error(y, y_fcst)
                    metrics_list.append(err.reset_index(drop=True))
                metrics_df = pd.concat(metrics_list, axis=1)
                metrics_df.columns = np.arange(len(metrics_df.columns))
                target_metrics[target] = metrics_df.T
            metrics_dict[key] = pd.concat(target_metrics, axis=1)
        scores_df = pd.concat(metrics_dict, axis=1)
        return scores_df

    def backtest(self, data, predictors=None, metrics=None, agg=None, fh=1, start=0.5, stride=1, progress=True):
        if metrics is None:
            metrics = [mae]
        hfcst = self.historical_forecasts(data, predictors, fh=fh, start=start, stride=stride, progress=progress)
        metrics_dict = dict()
        for key in self.get_keys(data):
            target_metrics = dict()
            for target in self.target_cols:
                metrics_list = list()
                for i, fcst in enumerate(hfcst):
                    step_metrics = dict()
                    y = self.get_indexed_series(data, key, target)
                    y_fcst = self.get_indexed_series(fcst, key, target + "_fcst")
                    y, y_fcst = self.af_trim(y, y_fcst)
                    for metric in metrics:
                        step_metrics[metric.__name__] = metric(y, y_fcst)
                    metrics_list.append(pd.Series(step_metrics))
                metrics_df = pd.DataFrame(metrics_list)
                target_metrics[target] = metrics_df
            metrics_dict[key] = pd.concat(target_metrics, axis=1)
        scores_df = pd.concat(metrics_dict, axis=1)
        if agg is not None:
            return agg(scores_df)
        return scores_df

    def plot_fcst(self, data, fcst, key, target, style="-"):
        target_fcst = target + "_fcst"
        target_lower = target + "_lower"
        target_upper = target + "_upper"
        f, ax = plt.subplots()
        y = self.get_indexed_series(data, key, target)
        y_fcst = self.get_indexed_series(fcst, key, target_fcst)
        y.plot(ax=ax, style=style)
        y_fcst.plot(ax=ax, style=style)
        if target_lower in fcst.columns and target_upper in fcst.columns:
            y_lower = self.get_indexed_series(fcst, key, target_lower)
            y_upper = self.get_indexed_series(fcst, key, target_upper)
            ax.fill_between(x=y_fcst.index, y1=y_lower, y2=y_upper, alpha=0.8, color="lightblue")
        plt.legend()

    def plot_forecasts(self, data, fcst):
        interact(
            self.plot_fcst,
            data=widgets.fixed(data),
            fcst=widgets.fixed(fcst),
            key=self.get_keys(data),
            target=self.target_cols,
            style=["-", "."])
