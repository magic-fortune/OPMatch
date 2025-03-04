python train_opmatch.py \
--exp "pct_lab6_re" \
--dataset_name "Pancreas_CT" \
--conf_thresh 0.85 \
--label_num 6 \
--max_iterations 12000 \
--base_lr 0.001


python test.py --model "pct_lab6" --dataset_name "Pancreas_CT"
