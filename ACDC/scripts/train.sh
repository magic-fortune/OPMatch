#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['acdc']
# method: ['unimatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['1', '3', '7']
dataset='acdc'
method='train_opmatch'
exp='unet'


ratio=0.5
note='train_opmatch'
config=configs/$dataset.yaml


labeled_id_path=splits/$dataset/3/labeled.txt
unlabeled_id_path=splits/$dataset/3/unlabeled.txt
save_path=exp_re/$dataset/${method}_$note/$exp/3
mkdir -p $save_path
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path   --ratio=$ratio --scale=1e-2 \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log \


labeled_id_path=splits/$dataset/7/labeled.txt
unlabeled_id_path=splits/$dataset/7/unlabeled.txt
save_path=exp_re/$dataset/${method}_$note/$exp/7
mkdir -p $save_path
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path   --ratio=$ratio --scale=1e-3 \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log \




