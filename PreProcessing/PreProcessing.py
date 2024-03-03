import pandas as pd
import numpy as np


class PreProcessing:
    def split_dataset(self, dataset, valid_ratio):
        valid_indices = np.random.rand(len(dataset)) < valid_ratio
        return dataset[~valid_indices], dataset[valid_indices]

    def encode(self, dataset, feature):
        ordering = pd.DataFrame()
        ordering['val'] = dataset[feature].unique()
        ordering.index = ordering.values
        ordering['spmean'] = dataset[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
        ordering = ordering.sort_values('spmean')
        ordering['ordering'] = range(1, ordering.shape[0] + 1)
        ordering = ordering['ordering'].to_dict()

        for cat, o in ordering.items():
            dataset.loc[dataset[feature] == cat, feature + '_E'] = o
