import pandas as pd
from tstools.spike_detection import detect_spikes
from skimage.restoration import denoise_tv_chambolle


def denoise(y, weight=0.01):
    y_denoised = denoise_tv_chambolle(y.to_numpy(), weight=weight)
    y_denoised = pd.Series(y_denoised, index=y.index)
    return y_denoised


def detect_jumps(y, window=5, alpha=3, delta=1, weight=0.01):
    y_diff = y_diff = y.diff().bfill()
    y_diff_denoised = denoise(y_diff, weight=weight)
    jumps = detect_spikes(y_diff_denoised, window=window, alpha=alpha, delta=delta)
    jumps.name = "jump"
    return jumps
