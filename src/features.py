import numpy as np
from scipy.signal import welch

def bandpower(data, sf, band, window_sec=None):
    band = np.array(band)
    low, high = band
    freqs, psd = welch(data, sf, nperseg=window_sec*sf if window_sec else None)
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    return np.trapz(psd[idx_band], freqs[idx_band])

def extract_features(data_dict):
    signals = data_dict["signals"]   # shape: (trials, 32, 8064)
    labels = data_dict["labels"]     # shape: (trials, 2) → [valence, arousal]
    sf = 128
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50)
    }
    features = []
    for trial in signals:
        trial_features = []
        for ch in range(trial.shape[0]):
            for band in bands.values():
                bp = bandpower(trial[ch], sf, band)
                trial_features.append(bp)
        features.append(trial_features)
    X = np.array(features)
    y = labels[:, 0]  # example: only valence classification
    return X, y
