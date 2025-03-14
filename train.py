import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression


from utils import set_seed, preprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cross_val", action="store_true")
    parser.add_argument("--n_splits", default=5, type=int)

    parser.add_argument("--missing", default=None, choices=["mode", "drop"])
    parser.add_argument("--merge_edu", action="store_true")
    parser.add_argument("--merge_marital", action="store_true")
    parser.add_argument("--merge_gain_loss", action="store_true")
    parser.add_argument("--merge_race", action="store_true")
    parser.add_argument("--merge_country", action="store_true")
    parser.add_argument("--merge_workclass", action="store_true")
    parser.add_argument("--trans_fnlwgt", default=None, choices=["log"])

    parser.add_argument("--train_or_test", default="train", type=str)

    args = parser.parse_args()
    return args

def onehot_or_label(cols):
    # numeric: 0
    # one_hot : 1
    # label: 2
    output_col = "income"
    cfg = {
        "workclass": 1,
        "education": 1,
        "education-num": 1,
        "marital-status": 1,
        "occupation": 1,
        "relationship": 1,
        "race": 1,
        "sex": 1,
        "native-country": 1,
        "age": 0, 
        "fnlwgt": 0, 
        "hours-per-week": 0, 
        "income": 0,
        "capital": 0,
        "capital-gain": 0,
        "capital-loss": 0,
    }
    num_col = []
    cat_col = []
    label_col = []
    for col in cols:
        if col == output_col:
            continue
        if cfg[col] == 0:
            num_col.append(col)
        elif cfg[col] == 1:
            cat_col.append(col)
        else:
            label_col.append(col)
    return num_col, cat_col, label_col

if __name__ == '__main__':
    args = parse_args()
    # set_seed(args.seed)
    train_data_path = os.path.join(args.data_dir, "train.csv")
    test_data_path = os.path.join(args.data_dir, "test.csv")

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    df_train = preprocess(train_data, args)
    args.train_or_test = "test"
    df_test = preprocess(test_data, args)

    print("*"*20, "Encode data", "*"*20)
    cols = df_train.columns.tolist()
    num_cols, cat_cols, label_cols = onehot_or_label(cols)

    print("Numerical features: ", num_cols)
    print("Catagorical features: ", cat_cols)
    print("Label features: ", label_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(), cat_cols),
            ("label", LabelEncoder(), label_cols),
            ("numeric", "passthrough", num_cols),
        ],
        remainder="passthrough"
    )    
    X_train_all = df_train.drop(columns="income")
    Y_train_all = df_train["income"]

    X_test = df_test.drop(columns="income")
    Y_test = df_test["income"]

    print(X_train_all.shape)
    X_train_processed = preprocessor.fit_transform(X_train_all)
    X_test_processed = preprocessor.transform(X_test)
    print(X_train_processed.shape)

    if args.cross_val:
        print("*"*20, f"Using {args.n_splits}-fold cross validations", "*"*20)
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        for i, (train_index, val_index) in enumerate(skf.split(X_train_all, Y_train_all)):
            print("-"*20, f"Fold {i}", "-"*20)
            print(f"{len(train_index)} training data, {len(val_index)} validation data")
            X_train = X_train_all.iloc[train_index]
            Y_train = Y_train_all.iloc[train_index]
            X_val = X_train_all.iloc[val_index]
            Y_val = Y_train_all.iloc[val_index]
    else:
        print("*"*20, "Not using cross validations", "*"*20)
        X_train = X_train_all
        Y_train = Y_train_all