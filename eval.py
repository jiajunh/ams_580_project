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
    data_path = os.path.join(args.data_dir, "test.csv")

    data = pd.read_csv(data_path)
    print("Missing data", data.isnull().sum())
    print("Data shape", data.shape)