import os
import argparse
import numpy as np
import pandas as pd

from utils import set_seed, preprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    data_path = os.path.join(args.data_dir, "train.csv")

    data = pd.read_csv(data_path)
    cols = data.columns.tolist()
    print("Colomn names: ", cols)
    print("Data types: ", data.dtypes)
    print("Missing data", data.isnull().sum())
    print("Data shape", data.shape)
    # print(data[cols[1]].value_counts()['?'])