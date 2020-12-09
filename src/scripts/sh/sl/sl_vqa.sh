#!/usr/bin/env bash
python src/train/launch_train.py -task "policy" -dataset "vqa" -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -out_path "output/vqa_policy_newvocab" -emb_size 512 -hidden_size 1024 -bs 512 -ep 20 -num_workers 6 -fusion "average" -condition_answer "after_fusion" -min_data 0 -device_id 0

python src/train/launch_train.py -task "policy" -dataset "vqa" -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -out_path "output/vqa_policy_smallvocab" -emb_size 512 -hidden_size 1024 -bs 512 -ep 20 -num_workers 6 -fusion "average" -condition_answer "after_fusion" -min_data 1 -device_id 1

python src/train/launch_train.py -task "lm" -dataset "vqa" -model "lstm" -num_layers 1 -emb_size 512 -hidden_size 512 -p_drop 0.1 -lr 0.001 -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -out_path "output/vqa_lm_model_newvocab" -bs 512 -ep 20 -num_workers 6 -min_data 0 -device_id 2

python src/train/launch_train.py -task "lm" -dataset "vqa" -model "lstm" -num_layers 1 -emb_size 512 -hidden_size 512 -p_drop 0.1 -lr 0.001 -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -out_path "output/vqa_lm_model_small_vocab" -bs 512 -ep 50 -num_workers 6 -min_data 1 -device_id 3


