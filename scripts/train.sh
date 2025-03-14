set -ex

python train.py \
    --data_dir "./data/" \
    --seed 42 \
    --train_or_test "train" \
    --missing "mode" \
    --trans_fnlwgt "log" \
    --merge_edu \
    --merge_marital \
    --merge_gain_loss \
    --merge_race \