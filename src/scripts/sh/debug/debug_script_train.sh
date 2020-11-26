#!/usr/bin/env bash

#echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")-------------------------- train LM on CLEVR ---------------------------------------------------------------------------------------------------"
#python src/train/launch_train.py -task "lm" -dataset "clevr" -model "lstm" -num_layers 1 -emb_size 16 -hidden_size 16 -p_drop 0.1 -lr 0.001 -data_path "data" -out_path "output/temp" -bs 6 -ep 1 -num_workers 6
#echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")-------------------------- train LM on VQA ---------------------------------------------------------------------------------------------------"
#python src/train/launch_train.py -task "lm" -dataset "vqa" -model "lstm" -num_layers 1 -emb_size 16 -hidden_size 16 -p_drop 0.1 -lr 0.001 -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -out_path "output/temp" -bs 512 -ep 1 -num_workers 6
#echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")-------------------------- train policy on CLEVR ---------------------------------------------------------------------------------------------------"
#python src/train/launch_train.py -task "policy" -dataset "clevr" -data_path "data" -out_path "output/temp" -emb_size 16 -hidden_size 16 -bs 512 -ep 1 -num_workers 6 -fusion "cat" -max_samples 5000
#echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")-------------------------- train policy on CLEVR - condition_answer  ---------------------------------------------------------------------------------------------------"
#python src/train/launch_train.py -task "policy" -dataset "clevr" -data_path "data" -out_path "output/temp" -emb_size 16 -hidden_size 16 -bs 512 -ep 1 -num_workers 6 -fusion "cat" -condition_answer "after_fusion" -max_samples 5000
#echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")-------------------------- train policy on VQA  ---------------------------------------------------------------------------------------------------"
#python src/train/launch_train.py -task "policy" -dataset "vqa" -data_path "data" -out_path "output/temp" -emb_size 16 -hidden_size 16 -bs 512 -ep 1 -num_workers 6 -fusion "average"
#echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")-------------------------- train policy on VQA - condition_answer  ---------------------------------------------------------------------------------"
#python src/train/launch_train.py -task "policy" -dataset "vqa" -data_path "data/vqa-v2" --features_path "data/vqa-v2/coco_trainval.lmdb" -out_path "output/temp" -emb_size 16 -hidden_size 16 -bs 512 -ep 1 -num_workers 6 -fusion "average" -condition_answer "after_fusion"
#echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")----------------------END-------------------------------------------------------------------"

