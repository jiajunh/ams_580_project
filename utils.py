import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def preprocess(df, args):
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
        | education-num (Identical to education, just drop it)
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

    missing_cols = ["workclass", "occupation", "native-country"]
    categorical = [
        "workclass", 
        "education", 
        "marital-status", 
        "occupation", 
        "relationship", 
        "race", 
        "sex", 
        "native-country"
    ]
    numerical = [
        "age",
        "fnlwgt",
        "education",
        "capital-gain",
        "capital-loss",
        "capital",
        "hours-per-week",
    ]
    missing_mode = {
        "workclass": "Private",
        "occupation": "Prof-specialty",
        "native-country": "United-States",
    }

    merge_workclass_cfg = {
        "Private": "Private",
        "Self-emp-not-inc": "Self-emp-not-inc",
        "Self-emp-inc": "Self-emp-inc", 
        "Federal-gov": "Federal-gov",
        "Local-gov": "Local-gov", 
        "State-gov": "State-gov",
        "Without-pay": "no_income", 
        "Never-worked": "no_income",
    }

    merge_edu_cfg = {
        "Preschool": "before_high_school",
        "1st-4th": "before_high_school",
        "5th-6th": "before_high_school",
        "7th-8th": "before_high_school",
        "9th": "before_high_school",
        "10th": "high_school",
        "11th": "high_school",
        "12th": "high_school",
        "HS-grad": "HS_grad",
        "Prof-school": "Prof_school",
        "Assoc-acdm": "Assoc_acdm",
        "Assoc-voc": "Assoc_voc",
        "Some-college": "Some_college",
        "Bachelors": "Bachelors",
        "Masters": "Masters",
        "Doctorate": "Doctorate",
    }

    merge_marital_cfg = {
        "Never-married": "single",
        "Separated": "single",
        "Divorced": "single",
        "Widowed": "single",
        "Married-civ-spouse": "married",
        "Married-spouse-absent": "married", 
        "Married-AF-spouse": "married",
    }

    merge_race_cfg = {
        "White": "white", 
        "Black": "black",
        "Asian-Pac-Islander": "others",
        "Amer-Indian-Eskimo": "others", 
        "Other": "others",
    }

    if args.train_or_test == "train":
        print("*"*20, "Preprocess train data", "*"*20)
    else:
        print("*"*20, "Preprocess test data", "*"*20)

    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})
    df = df.drop(columns="education-num")

    if args.missing:
        if args.missing == "mode" or args.train_or_test == "test":
            for col in missing_cols:
                print(f"{col}: Replacing missing values with mode")
                df[col] = df[col].map(lambda x: missing_mode[col] if x == "?" else x)
        elif args.missing == "drop" and args.train_or_test == "train":
            for col in missing_cols:
                print(f"{col}: Dropping missing values")
                df = df[df[col] != "?"]

    if args.merge_workclass:
        print("workclass: Merging workclass")
        df["workclass"] = df["workclass"].map(lambda x: merge_workclass_cfg[x])

    if args.merge_edu:
        print("education: Merging education")
        df["education"] = df["education"].map(lambda x: merge_edu_cfg[x])

    if args.merge_marital:
        print("marital_status: Merge marital")
        df["marital-status"] = df["marital-status"].map(lambda x: merge_marital_cfg[x])
    
    if args.merge_race:
        print("race: Merge race")
        df["race"] = df["race"].map(lambda x: merge_race_cfg[x])

    if args.merge_country:
        print("country: Merge country")
        df["native-country"] = df["native-country"].map(
            lambda x: "United-States" if x == "United-States" else "others")

    if args.merge_gain_loss:
        print("capital_gain: Drop capital-gain column")
        print("capital_loss: Drop capital-loss column")
        print("capital: Create capital column = capital-gain - capital-loss")
        df["capital"] = df["capital-gain"] - df["capital-loss"]
        df = df.drop(columns=["capital-gain", "capital-loss"])

    if args.trans_fnlwgt:
        if args.trans_fnlwgt == "log":
            print("fnlwgt: Log transform")
            df["fnlwgt"] = df["fnlwgt"].map(np.log)

    return df
