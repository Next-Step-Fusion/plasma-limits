import pickle
import importlib
from pathlib import Path


def load_weights(weights_files):
    weights_paths = [Path(f) for f in weights_files]
    weights_list = []
    for f in weights_paths:
        if f.is_file():
            with open(f, 'rb') as bf:
                weights_list.append(pickle.load(bf))
    return weights_list
