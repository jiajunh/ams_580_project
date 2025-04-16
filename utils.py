import os
import pickle
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


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
        "education-num",
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
        "?": "?",
    }

    merge_edu_cfg = {
        "Preschool": "before_high_school",
        "1st-4th": "before_high_school",
        "5th-6th": "before_high_school",
        "7th-8th": "before_high_school",
        "9th": "high_school",
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

    if args.pre_data == "train":
        print("*"*20, "Preprocess train data", "*"*20)
    else:
        print("*"*20, "Preprocess test data", "*"*20)

    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})
    df = df.drop(columns="education-num")

    if args.missing:
        if args.missing == "mode":
            for col in missing_cols:
                print(f"{col}: Replacing missing values with mode")
                df[col] = df[col].map(lambda x: missing_mode[col] if x == "?" else x)
        elif args.missing == "drop":
            if args.train_or_test == "train":
                for col in missing_cols:
                    print(f"{col}: Dropping missing values")
                    df = df[df[col] != "?"]
            else:
                for col in missing_cols:
                    print(f"{col}: Replacing missing values with mode")
                    df[col] = df[col].map(lambda x: missing_mode[col] if x == "?" else x)

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

    if args.trans_capital:
        if args.trans_fnlwgt == "log":
            print("capital: Log transform")
            if "capital" in df.columns.tolist():
                df["capital"] = df["capital"].map(lambda x: np.log(x - np.min(df["capital"]) + 1))
            if "capital-gain" in df.columns.tolist():
                df["capital-gain"] = df["capital-gain"].map(lambda x: np.log(x - np.min(df["capital-gain"]) + 1))
            if "capital-loss" in df.columns.tolist():
                df["capital-loss"] = df["capital-loss"].map(lambda x: np.log(x - np.min(df["capital-loss"]) + 1))

    return df


def remove_outliers(df, args):
    print("-"*20, "Start to remove outliers", "-"*20)
    feature_cfg = {
        "age": {"mu": 38.560981227686284, "std": 13.622763128883086},
        "fnlwgt": {"mu": 11.984648125210024, "std": 0.630349936466534},
        # "education-num": {},
        # "capital-gain": {},
        # "capital-loss": {},
        "capital": {"mu": 8.440271484370996, "std": 0.3762841106207502},
        "hours-per-week": {"mu": 40.45266192581203, "std": 12.310561060717218},
    }
    cols = df.columns.tolist()
    for col in cols:
        if col in feature_cfg.keys():
            prev_length = df.shape[0]
            if args.outlier_strategy == "sigma":
                mu = feature_cfg[col]["mu"] 
                std = feature_cfg[col]["std"] # 
                lb = mu - 3.6 * std
                ub = mu + 3.6 * std
            elif args.outlier_strategy == "iqr":
                q1 = df[col].quantile(0.25)
                mu = df[col].quantile(0.5)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lb = q1 - 5 * iqr
                ub = q1 + 5 * iqr
            df_outlier = df[(df[col] < lb) | (df[col] > ub)]
            df = df[(df[col] >= lb) & (df[col] <= ub)]
            print(f"Col: {col}, filter {prev_length-df.shape[0]} data")
    return df, df_outlier

def compute_scores(y, pred):
    cm =  confusion_matrix(y, pred)
    TP = cm[0,0]
    FP = cm[1,0]
    FN = cm[0,1]
    TN = cm[1,1]
    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / (TP + FN)
    specificity = 1.0 * TN / (TN + FP) 
    accuracy = 1.0 * (TP + TN) / (TP + TN + FP + FN)
    # print(f"Confusion Matrix: TP={TP}, FN={FN}, FP={FP}, TN={TN}")
    print(f"Accuracy: {accuracy:.4f}, Sensitivity: {recall:.4f}, Specificity: {specificity:.4f}")
    return cm, precision, recall, specificity, accuracy


def update_best_model(model_cfg, model_name, model, scores, args, save=True):
    cm, precision, recall, specificity, accuracy = scores
    if model_cfg[model_name] is None or model_cfg[model_name]["acc"] < accuracy:
        model_cfg[model_name] = {
            "model": model,
            "acc": accuracy,
            "confusion_matrix": cm,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
        }
        print(f"Model updates: current {model_name} is the best model, acc={accuracy:.4f}")
        if save:
            save_model(model_name, model, accuracy, args)
    return model_cfg


def save_model(model_name, model, acc, args):
    model_suffix = {
        "logistic_regression": ".pkl",
        "neural_network": ".pt",
        "xgboost": ".pkl",
        "random_forest": ".pkl",
        "svm": ".pkl",
    }
    model_path = os.path.join(args.model_dir, model_name)
    model_path = model_path +  f"_{int(acc*10000)}" + model_suffix[model_name]
    if model_suffix[model_name]:
        if model_suffix[model_name] in [".pkl"]:
            with open(model_path, "wb") as file:
                pickle.dump(model, file)
        elif model_suffix[model_name] in [".pt"]:
            torch.save(model.state_dict(), model_path)

    print(f"{model_name} model has saved to {model_path}")


def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def load_nn_model(model_path, args):
    model = args.nn_model
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def get_saved_model(best_models, args):
    model_paths = os.listdir(args.model_dir)
    for path in model_paths:
        if path.split(".")[-1] in ["pkl"]:
            temp_name = path.split(".")[0]
            temp_model_name = temp_name[0:-5]
            temp_acc = int(temp_name[-4:])
            if best_models[temp_model_name] is None or best_models[temp_model_name]["acc"] < temp_acc:
                best_models[temp_model_name] = {
                    "model": load_model(os.path.join(args.model_dir, path)),
                    "acc": 1.0 * temp_acc / 10000,
                    "confusion_matrix": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "specificity": 0.0,
                }
        elif path.split(".")[-1] in ["pt"]:
            temp_name = path.split(".")[0]
            temp_model_name = temp_name[0:-5]
            temp_acc = int(temp_name[-4:])
            if best_models[temp_model_name] is None or best_models[temp_model_name]["acc"] < temp_acc:
                best_models[temp_model_name] = {
                    "model": load_nn_model(os.path.join(args.model_dir, path), args),
                    "acc": 1.0 * temp_acc / 10000,
                    "confusion_matrix": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "specificity": 0.0,
                }

    return best_models