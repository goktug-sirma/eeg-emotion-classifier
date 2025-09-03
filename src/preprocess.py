import numpy as np
import os
import pickle

def load_deap(path, subjects=None, trials=None):
    files = [f for f in os.listdir(path) if f.endswith(".dat")]
    data = []
    labels = []
    for f in sorted(files):
        with open(os.path.join(path, f), "rb") as file:
            obj = pickle.load(file, encoding="latin1")
            eeg = obj["data"]    # shape: (40, 40, 8064)
            lbl = obj["labels"]  # shape: (40, 4)
            if trials:
                eeg = eeg[:trials]
                lbl = lbl[:trials]
            for i in range(len(eeg)):
                data.append(eeg[i, :32, :])  # only first 32 EEG channels
                labels.append(lbl[i, :2])    # valence, arousal
    return np.array(data), np.array(labels)

def preprocess_data(path):
    X, y = load_deap(path)
    return {"signals": X, "labels": y}
