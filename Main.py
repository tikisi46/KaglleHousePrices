import numpy as np
import pandas as pd
import tensorflow as tf

import PreProcessing

train_file_path = r"C:\Users\ishii\github\KaglleHousePrices\input\house-prices-advanced-regression-techniques\train.csv"


class Main:
    label = 'SalePrice'

    train = pd.read_csv(train_file_path)
    train = train.drop('Id', axis=1)

    quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
    quantitative.remove('SalePrice')
    qualitative = [f for f in train.columns if train.dtypes[f] == 'object']

    preProcessing = PreProcessing()

    qual_encoded = []
    for q in qualitative:
        preProcessing.encode(train, q)
        qual_encoded.append(q + '_E')

    train_ds, valid_ds = preProcessing.split_dataset(train, 0.2)


