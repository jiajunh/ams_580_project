import os
import time
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


from utils import set_seed, preprocess, compute_scores, update_best_model, get_saved_model
from model import train_logistic_model, train_svm_model, train_random_forest, train_neural_network, eval_nn_model, Net, train_xgboost

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--model_dir", default="./models/", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cross_val", action="store_true")
    parser.add_argument("--n_splits", default=5, type=int)

    # SMOTE even get worse result, easier to overfit
    parser.add_argument("--use_smote", action="store_true")
    parser.add_argument("--remove_outlier", action="store_true")

    parser.add_argument("--missing", default=None, choices=["mode", "drop"])
    parser.add_argument("--merge_edu", action="store_true")
    parser.add_argument("--merge_marital", action="store_true")
    parser.add_argument("--merge_gain_loss", action="store_true")
    parser.add_argument("--merge_race", action="store_true")
    parser.add_argument("--merge_country", action="store_true")
    parser.add_argument("--merge_workclass", action="store_true")
    parser.add_argument("--trans_fnlwgt", default=None, choices=["log"])

    parser.add_argument("--use_logitic_regression", action="store_true")
    parser.add_argument("--use_neural_network", action="store_true")
    parser.add_argument("--use_xgboost", action="store_true")
    parser.add_argument("--use_random_forest", action="store_true")
    parser.add_argument("--use_svm", action="store_true")

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
    set_seed(args.seed)
    train_data_path = os.path.join(args.data_dir, "train.csv")
    test_data_path = os.path.join(args.data_dir, "test.csv")

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    args.pre_data = "train"
    df_train = preprocess(train_data, args)
    args.pre_data = "test"
    df_test = preprocess(test_data, args)

    print("*"*20, "Encode data", "*"*20)
    cols = df_train.columns.tolist()
    num_cols, cat_cols, label_cols = onehot_or_label(cols)

    print("Numerical features: ", num_cols)
    print("Catagorical features: ", cat_cols)
    print("Label features: ", label_cols)

    use_scale = {
        "logistic_regression": True,
        "neural_network": True,
        "xgboost": False,
        "random_forest": False,
        "svm": True,
    }

    preprocessor_scale = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(), cat_cols),
            ("label", LabelEncoder(), label_cols),
            ('scaler', StandardScaler(), num_cols),
            # ("numeric", "passthrough", num_cols),
        ],
        remainder="passthrough"
    )
    preprocessor_pass = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(), cat_cols),
            ("label", LabelEncoder(), label_cols),
            # ('scaler', StandardScaler(), num_cols),
            ("numeric", "passthrough", num_cols),
        ],
        remainder="passthrough"
    )

    X_train_all = df_train.drop(columns="income")
    Y_train_all = df_train["income"]

    X_test = df_test.drop(columns="income")
    Y_test = df_test["income"]

    X_train_scale = pd.DataFrame(preprocessor_scale.fit_transform(X_train_all).toarray())
    X_train_pass = pd.DataFrame(preprocessor_pass.fit_transform(X_train_all).toarray())
    X_test_scale = pd.DataFrame(preprocessor_scale.transform(X_test).toarray())
    X_test_pass = pd.DataFrame(preprocessor_pass.transform(X_test).toarray())

    Y_train_scale = Y_train_all
    Y_train_pass = Y_train_all

    if args.use_smote:
        print("-"*20, "Using SMOTE", "-"*20)
        print(f"Original training data, total:{Y_train_all.shape[0]}, 1:{sum(Y_train_all==1)}, 0:{sum(Y_train_all==0)}")
        sm = SMOTE(random_state=args.seed)
        X_train_scale, Y_train_scale = sm.fit_resample(X_train_scale, Y_train_all)
        sm = SMOTE(random_state=args.seed)
        X_train_pass, Y_train_pass = sm.fit_resample(X_train_pass, Y_train_all)
        print(f"After SMOTE, total:{Y_train_scale.shape[0]}, 1:{sum(Y_train_scale==1)}, 0:{sum(Y_train_scale==0)}")

    args.input_dim = X_train_scale.shape[1]
    args.nn_model = Net(input_size=args.input_dim)

    best_models = {
        "logistic_regression": None,
        "neural_network": None,
        "xgboost": None,
        "random_forest": None,
        "svm": None,
    }
    get_saved_model(best_models, args)

    if args.train_or_test == "train":
        print("*"*20, f"Training Step", "*"*20)
        if args.cross_val:
            print("*"*20, f"Using {args.n_splits}-fold cross validations", "*"*20)
            skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
            for i, (train_index, val_index) in enumerate(skf.split(X_train_scale, Y_train_scale)):
                print("-"*20, f"Fold {i}", "-"*20)
                print(f"{len(X_train_scale)} training data, {len(val_index)} validation data")
                
                X_train_s = X_train_scale.iloc[train_index]
                X_train_p = X_train_pass.iloc[train_index]
                Y_train_s = Y_train_scale.iloc[train_index]
                Y_train_p = Y_train_pass.iloc[train_index]
                
                X_val_s = X_train_scale.iloc[val_index]
                X_val_p = X_train_pass.iloc[val_index]
                Y_val_s = Y_train_scale.iloc[val_index]
                Y_val_p = Y_train_pass.iloc[val_index]

                # Logistic Regression
                if args.use_logitic_regression:
                    print("-"*20, "Training Logistic Model", "-"*20)
                    start_time = time.time()
                    X_train, X_val, Y_train, Y_val = (X_train_s, X_val_s, Y_train_s, Y_val_s) if use_scale["logistic_regression"] else (X_train_p, X_val_p, Y_train_p, Y_val_p)
                    scale_or_path = "scale" if use_scale["logistic_regression"] else "path"
                    lr_model = train_logistic_model(X_train, Y_train, args)
                    end_time = time.time()
                    print(f"Train logistic_regression model with {scale_or_path} data, using {end_time - start_time :.3f}s")
                    y_pred = lr_model.predict(X_val)
                    scores = compute_scores(Y_val, y_pred)
                    best_models = update_best_model(best_models, "logistic_regression", lr_model, scores, args)

                # SVM
                if args.use_svm:
                    print("-"*20, "Training SVM Model", "-"*20)
                    start_time = time.time()
                    X_train, X_val, Y_train, Y_val = (X_train_s, X_val_s, Y_train_s, Y_val_s) if use_scale["svm"] else (X_train_p, X_val_p, Y_train_p, Y_val_p)
                    scale_or_path = "scale" if use_scale["svm"] else "path"
                    svm_model = train_svm_model(X_train, Y_train, args)
                    end_time = time.time()
                    print(f"Train svm model with {scale_or_path} data, using {end_time - start_time :.3f}s")
                    y_pred = svm_model.predict(X_val)
                    scores = compute_scores(Y_val, y_pred)
                    best_models = update_best_model(best_models, "svm", svm_model, scores, args)

                # Random forest
                if args.use_random_forest:
                    print("-"*20, "Training Random Forest", "-"*20)
                    start_time = time.time()
                    X_train, X_val, Y_train, Y_val = (X_train_s, X_val_s, Y_train_s, Y_val_s) if use_scale["random_forest"] else (X_train_p, X_val_p, Y_train_p, Y_val_p)                    
                    scale_or_path = "scale" if use_scale["random_forest"] else "path"
                    rf_model = train_random_forest(X_train, Y_train, args)
                    end_time = time.time()
                    print(f"Train random forest with {scale_or_path} data, using {end_time - start_time :.3f}s")
                    y_pred = rf_model.predict(X_val)
                    scores = compute_scores(Y_val, y_pred)
                    best_models = update_best_model(best_models, "random_forest", rf_model, scores, args)

                if args.use_neural_network:
                    print("-"*20, "Training Neural Network", "-"*20)
                    start_time = time.time()
                    X_train, X_val, Y_train, Y_val = (X_train_s, X_val_s, Y_train_s, Y_val_s) if use_scale["neural_network"] else (X_train_p, X_val_p, Y_train_p, Y_val_p)                    
                    scale_or_path = "scale" if use_scale["neural_network"] else "path"
                    nn_model = train_neural_network(X_train, Y_train, X_val, Y_val, args)
                    end_time = time.time()
                    print(f"Train neural network with {scale_or_path} data, using {end_time - start_time :.3f}s")
                    y, pred = eval_nn_model(nn_model, X_val, Y_val)
                    scores = compute_scores(y, pred)
                    best_models = update_best_model(best_models, "neural_network", nn_model, scores, args)
                
                if args.use_xgboost:
                    print("-"*20, "Training XGBoost", "-"*20)
                    start_time = time.time()
                    X_train, X_val, Y_train, Y_val = (X_train_s, X_val_s, Y_train_s, Y_val_s) if use_scale["xgboost"] else (X_train_p, X_val_p, Y_train_p, Y_val_p)             
                    scale_or_path = "scale" if use_scale["xgboost"] else "path"
                    xgboost_model = train_xgboost(X_train, Y_train, args)
                    end_time = time.time()
                    print(f"Train xgboost with {scale_or_path} data, using {end_time - start_time :.3f}s")
                    y_pred = xgboost_model.predict(X_val)
                    scores = compute_scores(Y_val, y_pred)
                    best_models = update_best_model(best_models, "xgboost", xgboost_model, scores, args)
        else:
            print("*"*20, "Not using cross validations", "*"*20)

        
    print("*"*20, f"Testing Step", "*"*20)
    predictions = {}
    ensemble_weight = {
        "logistic_regression": 0.15,
        "neural_network": 0.2,
        "xgboost": 0.3,
        "random_forest": 0.15,
        "svm": 0.2,
    }
    for model_name in best_models.keys():
        if best_models[model_name] is not None:
            model = best_models[model_name]["model"]
            print("-"*20, f"{model_name}", "-"*20)

            X_test_data = X_test_scale if use_scale[model_name] else X_test_pass

            if model_name in ["logistic_regression", "random_forest", "svm", "xgboost"]:
                y_pred = model.predict(X_test_data)
                scores = compute_scores(Y_test, y_pred)
                predictions[model_name] = y_pred
            elif model_name in ["neural_network"]:
                y, pred = eval_nn_model(model, X_test_data, Y_test)
                scores = compute_scores(y, pred)
                predictions[model_name] = pred

    print("*"*20, "Final Ensemble Step", "*"*20)
    total_weight = np.sum([ensemble_weight[model_name] for model_name in predictions.keys()])
    ensemble_pred = np.zeros(Y_test.shape)
    final_weight = "Final ensembled model = "
    for model_name in predictions.keys():
        final_weight = final_weight + f"{ensemble_weight[model_name] / total_weight :.3f} * {model_name} + "
        ensemble_pred += ensemble_weight[model_name] / total_weight * predictions[model_name]
    ensemble_pred = (ensemble_pred > 0.5).astype(int)
    if final_weight:
        final_weight = final_weight[0:-2]
        print(final_weight)
    scores = compute_scores(Y_test, ensemble_pred)