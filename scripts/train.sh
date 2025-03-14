set -ex

python train.py \
    --data_dir "./data/" \
    --seed 42 \
    --train_or_test "test" \
    --missing "mode" \
    --trans_fnlwgt "log" \
    --merge_edu \
    --merge_marital \
    --merge_gain_loss \
    --merge_race \
    --merge_country \
    --merge_workclass \
    --cross_val \
    --use_logitic_regression \
    --use_neural_network \
    --use_xgboost \
    --use_random_forest \
    --use_svm \
    --use_scale \