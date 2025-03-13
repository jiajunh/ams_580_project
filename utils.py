import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        | education (Identical to education-num, just drop it)
            - type: str (Categorical)
            - dinstinct_values: 16
        | education-num 
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
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.drop(columns="education")
    df["income"] = np.where(df["income"]==">50K", 1, 0)


    # EDA
    df["fnlwgt"] = df["fnlwgt"].map(np.log)
    # df["fnlwgt"] = df["fnlwgt"].map(lambda x: 1/x)
    cols = df.columns.tolist()
    fig, ax = plt.subplots(4,4, figsize=(12, 8))
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.9, wspace=0.3, hspace=0.5)

    for i, col in enumerate(cols):
        r = i // 4
        c = i % 4
        if not isinstance(df[col][0], str):
            df[col].plot(kind="hist", ax=ax[r, c])
            ax[r, c].set_title(col)
        else:
            count = df[col].value_counts()
            # print(type(count), count.iloc[1])
            df[col].value_counts().plot.bar(ax=ax[r, c])
            ax[r, c].set_title(col)
    
    plt.show()
    

    return df
