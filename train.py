import os
import argparse
import numpy as np
import pandas as pd

from utils import set_seed, preprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--train_or_test", default="train", type=str)

    parser.add_argument("--missing", default=None, choices=["mode", "drop"])
    parser.add_argument("--merge_edu", action="store_true")
    parser.add_argument("--merge_marital", action="store_true")
    parser.add_argument("--merge_gain_loss", action="store_true")
    parser.add_argument("--merge_race", action="store_true")
    parser.add_argument("--trans_fnlwgt", default=None, choices=["log"])


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    data_path = os.path.join(args.data_dir, "train.csv")

    
    data = pd.read_csv(data_path)
    df = preprocess(data, args)
    # print(df.shape)

    