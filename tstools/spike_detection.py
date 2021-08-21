import numpy as np
import pandas as pd
from scipy.stats import median_absolute_deviation as mad
from skimage.restoration import denoise_tv_chambolle


def denoise(y, weight=0.01):
    y_denoised = denoise_tv_chambolle(y.to_numpy(), weight=weight)
    y_denoised = pd.Series(y_denoised, index=y.index)
    return y_denoised


def detect_spikes(y, window=5, alpha=3, delta=1):
    scaling_constant = 1.4826
    rolling_median = y.rolling(window, center=True).median()
    rolling_deviations = scaling_constant * y.rolling(window, center=True).apply(mad, raw=False)
    threshold = rolling_deviations.mean()
    spikes = (y - rolling_median).abs() > alpha*threshold
    y_df = pd.concat([y, spikes], axis=1).reset_index()
    y_df.columns = ["ds", "y", "spike"]

    spikes_loc = y_df.loc[y_df["spike"]].index
    intervals = []
    for index in spikes_loc:
        start = index - delta
        end = index + delta
        interval = set(np.arange(start, end + 1))
        intervals.append(interval)
    if intervals:
        spiky_neighborhood = list(set.union(*intervals))
        y_df.loc[spiky_neighborhood, "spike"] = True
    else:
        y_df["spike"] = False

    spikes = y_df.set_index("ds").loc[:, "spike"]
    return spikes


def detect_jumps(y, window=5, alpha=3, delta=1, weight=0.01):
    y_diff = y_diff = y.diff().bfill()
    y_diff_denoised = denoise(y_diff, weight=weight)
    jumps = detect_spikes(y_diff_denoised, window=window, alpha=alpha, delta=delta)
    jumps.name = "jump"
    return jumps


def remove_spikes(y, window=5, alpha=3, delta=1, drop=False):
    if drop:
        return y[~detect_spikes(y, window, alpha, delta)].copy()
    yc = y.copy()
    yc[detect_spikes(yc, window, alpha, delta)] = np.nan
    return yc
