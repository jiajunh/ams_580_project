set -ex

python train.py \
    --data_dir "./data/" \
    --seed 42 \
    --train_or_test "train" \
    --trans_fnlwgt "log" \
    --trans_capital "log" \
    --merge_edu \
    --merge_marital \
    --merge_gain_loss \
    --merge_race \
    --merge_country \
    --merge_workclass \
    --cross_val \
    --remove_outlier \
    --use_neural_network \
    --use_xgboost \
    --use_logitic_regression \
    --use_random_forest \
    --use_svm \
    # --missing "mode" \
    # --use_smote \

    