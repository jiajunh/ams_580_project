set -ex

python train.py \
    --data_dir "./data/" \
    --seed 215 \
    --train_or_test "train" \
    --trans_fnlwgt "log" \
    --trans_capital "log" \
    --cross_val \
    --remove_outlier \
    --merge_country \
    --merge_gain_loss \
    --use_xgboost \
    --use_neural_network \
    --use_logitic_regression \
    --use_random_forest \
    --use_svm \
    # --missing "mode" \
    # --use_smote \

    