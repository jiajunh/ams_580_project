import torch
import random
import numpy as np
import pandas as pd


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def preprocess(df):
    '''
    Data Structure
        | age
            - type: int
            - dinstinct_values: 73
        | workclass *(missing)
            - type: str (Categorical)
            - dinstinct_values: 9
        | fnlwgt
            - type: int
            - dinstinct_values: 18415
        | education
            - type: str (Categorical)
            - dinstinct_values: 16
        | education-num (Identical to education)
            - type: int
            - dinstinct_values: 16
        | marital-status
            - type: str (Categorical)
            - dinstinct_values: 7
        | occupation *(missing)
            - type: str (Categorical)
            - dinstinct_values: 15
        | relationship
            - type: str (Categorical)
            - dinstinct_values: 6
        | race
            - type: str (Categorical)
            - dinstinct_values: 5
        | sex
            - type: str (Binary)
            - dinstinct_values: 2
        | capital-gain
            - type: int
            - dinstinct_values: 116
        | capital-loss
            - type: int
            - dinstinct_values: 88
        | hours-per-week
            - type: int
            - dinstinct_values: 93
        | native-country *(missing)
            - type: str (Categorical)
            - dinstinct_values: 42
        | income
            - type: str (Binary)
            - dinstinct_values: 2
    '''
    cols = df.columns.tolist()
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    
    df = df.drop(columns="education")
    # for col in cols:
    #     print(col, type(df[col][0]), len(df[col].value_counts()))


    return df
