import numpy as np
import pandas as pd
import PreProcessing as pp
import tensorflow_decision_forests as tfdf

train_file_path = 'C:\Users\ishii\github\KaglleHousePrices\input\house-prices-advanced-regression-techniques\train.csv'


class Main:
    train = pd.read_csv(train_file_path)
    train = train.drop('Id', axis=1)
    train_ds, valid_ds = pp.split_dataset(train, 0.2)

    label = 'SalePrice'
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds, label=label, task=tfdf.keras.Task.REGRESSION)

