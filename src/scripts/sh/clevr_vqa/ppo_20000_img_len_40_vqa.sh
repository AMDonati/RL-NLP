#!/bin/bash
#echo "-------------------------- Pretrain ---------------------------------------------------------------------------------------------------"
#python src/scripts/run.py -max_len 40 -data_path "data" -out_path "output/RL/exp_clvr_vqa_20000img" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 100000 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 500 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -policy_path "output/SL_LSTM_32_64_vqa/model.pt"
#echo "-------------------------- top-k 10 ---------------------------------------------------------------------------------------------------"
#python src/scripts/run.py -max_len 40 -data_path "data" -out_path "output/RL/exp_clvr_vqa_20000img" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 100000 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 500 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -truncate_mode "top_k" -num_truncated 10 -grad_clip 1
#echo "-------------------------- top-k 20---------------------------------------------------------------------------------------------------"
#python src/scripts/run.py -max_len 40 -data_path "data" -out_path "output/RL/exp_clvr_vqa_20000img" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 100000 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 500 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -truncate_mode "top_k" -num_truncated 20 -grad_clip 1
#echo "-------------------------- sample-va 20---------------------------------------------------------------------------------------------------"
#python src/scripts/run.py -max_len 40 -data_path "data" -out_path "output/RL/exp_clvr_vqa_20000img" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 100000 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 500 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -truncate_mode "sample_va" -num_truncated 20 -grad_clip 1
#echo "-------------------------- sample-va 30---------------------------------------------------------------------------------------------------"
#python src/scripts/run.py -max_len 40 -data_path "data" -out_path "output/RL/exp_clvr_vqa_20000img" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 100000 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 500 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -truncate_mode "sample_va" -num_truncated 30 -grad_clip 1
echo "-------------------------- proba thr 1/V---------------------------------------------------------------------------------------------------"
python src/scripts/run.py -max_len 40 -data_path "data" -out_path "output/RL/exp_clvr_vqa_20000img" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 100000 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 500 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 1
echo "-------------------------- proba thr 0.05---------------------------------------------------------------------------------------------------"
python src/scripts/run.py -max_len 40 -data_path "data" -out_path "output/RL/exp_clvr_vqa_20000img" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 100000 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 500 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -truncate_mode "proba_thr" -p_th 0.05 -grad_clip 1
echo "-------------------------- top_p 0.5---------------------------------------------------------------------------------------------------"
python src/scripts/run.py -max_len 40 -data_path "data" -out_path "output/RL/exp_clvr_vqa_20000img" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 100000 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 500 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -truncate_mode "top_p" -top_p 0.5 -grad_clip 1
echo "-------------------------- top_p 0.8---------------------------------------------------------------------------------------------------"
python src/scripts/run.py -max_len 40 -data_path "data" -out_path "output/RL/exp_clvr_vqa_20000img" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 100000 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 500 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -truncate_mode "top_p" -top_p 0.8 -grad_clip 1

