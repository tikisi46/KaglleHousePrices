import numpy as np
import pandas as pd

train_file_path = 'C:\Users\ishii\github\KaglleHousePrices\input\house-prices-advanced-regression-techniques\train.csv'


class Main:
    train = pd.read_csv(train_file_path)
    train = train.drop('Id', axis = 1)

