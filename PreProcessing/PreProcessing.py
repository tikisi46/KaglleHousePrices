import pandas as pd
import numpy as np


class PreProcessing:
    def split_dataset(dataset, valid_ratio):
        valid_indices = np.random.rand(len(dataset)) < valid_ratio
        return dataset[~valid_indices], dataset[valid_indices]