#!/usr/bin/env bash
# VQA REWARD:
DATA_PATH="../vilbert-multi-task/data/datasets/VQA/"
FEATURES_PATH="../vilbert-multi-task/data/datasets/coco/features_100/COCO_trainval_resnext152_faster_rcnn_genome.lmdb/"
FILE_NAME="output/RL/test_before_merge_$(date +"%Y_%m_%d_%I_%M_%p").txt"

if [ -n "$1" ]
then
  DATA_PATH=$1
fi
if [ -n "$2" ]
then
  FEATURES_PATH=$2
fi

echo "-------------------------- Scratch ---------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env "vqa" -max_len 5 -data_path $DATA_PATH -out_path "output/RL/VQA_debug" -model "lstm" -update_every 7 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10 -debug "0,1" -lm_path "output/vqa_lm_model/model.pt" -reward "lv_norm" -num_episodes_test 5 -mask_answers 1 -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -features_path $FEATURES_PATH > $FILE_NAME
echo "------------------------- pretrain ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env "vqa" -max_len 5 -data_path $DATA_PATH -out_path "output/RL/VQA_debug" -model "lstm" -update_every 7 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10 -debug "0,1" -lm_path "output/vqa_lm_model/model.pt" -reward "lv_norm" -num_episodes_test 5 -mask_answers 1 -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -features_path $FEATURES_PATH -policy_path "output/SL_LSTM_32_64_vqa/model.pt" > $FILE_NAME
echo "------------------------- top k ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env "vqa" -max_len 5 -data_path $DATA_PATH -out_path "output/RL/VQA_debug" -model "lstm" -update_every 7 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10 -debug "0,1" -lm_path "output/vqa_lm_model/model.pt" -reward "lv_norm" -num_episodes_test 5 -mask_answers 1 -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -features_path $FEATURES_PATH  -truncate_mode "top_k" -num_truncated 20 > $FILE_NAME
echo "------------------------- sample_va ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env "vqa" -max_len 5 -data_path $DATA_PATH -out_path "output/RL/VQA_debug" -model "lstm" -update_every 7 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10 -debug "0,1" -lm_path "output/vqa_lm_model/model.pt" -reward "lv_norm" -num_episodes_test 5 -mask_answers 1 -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -features_path $FEATURES_PATH  -truncate_mode "sample_va" > $FILE_NAME
echo "------------------------- proba threshold ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env "vqa" -max_len 5 -data_path $DATA_PATH -out_path "output/RL/VQA_debug" -model "lstm" -update_every 7 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10 -debug "0,1" -lm_path "output/vqa_lm_model/model.pt" -reward "lv_norm" -num_episodes_test 5 -mask_answers 1 -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -features_path $FEATURES_PATH  -truncate_mode "proba_thr" > $FILE_NAME
echo "-------------------------  top p ---------------------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env "vqa" -max_len 5 -data_path $DATA_PATH -out_path "output/RL/VQA_debug" -model "lstm" -update_every 7 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10 -debug "0,1" -lm_path "output/vqa_lm_model/model.pt" -reward "lv_norm" -num_episodes_test 5 -mask_answers 1 -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -features_path $FEATURES_PATH  -truncate_mode "top_p" -top_p 0.8 > $FILE_NAME
echo "------------------------- sample_va + epsilon truncation ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env "vqa" -max_len 5 -data_path $DATA_PATH -out_path "output/RL/VQA_debug" -model "lstm" -update_every 7 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10 -debug "0,1" -lm_path "output/vqa_lm_model/model.pt" -reward "lv_norm" -num_episodes_test 5 -mask_answers 1 -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -features_path $FEATURES_PATH  -truncate_mode "sample_va"  -epsilon_truncated 0.2 2> $FILE_NAME
echo "------------------------- sample_va + GPT2 ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env "vqa" -max_len 5 -data_path $DATA_PATH -out_path "output/RL/VQA_debug" -model "lstm" -update_every 7 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10 -debug "0,1" -lm_path "gpt" -reward "lv_norm" -num_episodes_test 5 -mask_answers 1 -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -features_path $FEATURES_PATH  -truncate_mode "sample_va" 2> $FILE_NAME
