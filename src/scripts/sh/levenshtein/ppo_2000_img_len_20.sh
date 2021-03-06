#!/bin/bash
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_epstruncated" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 20 -debug "0,2000" -grad_clip 1 -num_questions 8 -truncate_mode "sample_va" -num_episodes_test 200 -epsilon_truncated 0.5 -eval_no_trunc 1 -train_seed 1
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_epstruncated" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 20 -debug "0,2000" -grad_clip 1 -num_questions 8 -truncate_mode "sample_va" -num_episodes_test 200 -epsilon_truncated 0.2 -eval_no_trunc 1 -train_seed 1
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_epstruncated" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 20 -debug "0,2000" -grad_clip 1 -num_questions 8 -truncate_mode "sample_va" -num_episodes_test 200 -epsilon_truncated 0.1 -eval_no_trunc 1 -train_seed 1
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_epstruncated" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 20 -debug "0,2000" -grad_clip 1 -num_questions 8 -truncate_mode "sample_va" -num_episodes_test 200 -eval_no_trunc 1 -train_seed 1

#python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_lmbonus" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -num_truncated 87 -debug "0,2000" -grad_clip 1 -num_questions 8 -lm_path "output/lm_model/model.pt" -alpha_logits 0.2 -num_episodes_test 200
#python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_lmbonus" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -num_truncated 87 -debug "0,2000" -grad_clip 1 -num_questions 8 -lm_path "output/lm_model/model.pt" -alpha_logits 0.1 -num_episodes_test 200
#python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_lmbonus" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -num_truncated 87 -debug "0,2000" -grad_clip 1 -num_questions 8 -lm_path "output/lm_model/model.pt" -alpha_logits 0.05 -num_episodes_test 200
#python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_lmbonus" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -num_truncated 87 -debug "0,2000" -grad_clip 1 -num_questions 8 -lm_path "output/lm_model/model.pt" -alpha_logits 0.5 -alpha_decay_rate 0.5 -num_episodes_test 200
#python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_lmbonus" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -num_truncated 87 -debug "0,2000" -grad_clip 1 -num_questions 8 -lm_path "output/lm_model/model.pt" -alpha_logits 0.2 -alpha_decay_rate 0.5 -num_episodes_test 200

python src/scripts/run.py -env "clevr" -max_len 20 -data_path "data" -out_path "output/RL/CLEVR_pretrain_baseline_Dec20" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -debug "0,2000" -num_questions 8 -num_episodes_test 200 -policy_path "output/SL_LSTM_32_64/model.pt" -reward "lv_norm" -condition_answer "none" -device_id 1


python src/scripts/run.py -env "clevr" -max_len 20 -data_path "data" -out_path "output/RL/CLEVR_pretrain_baseline_Dec20" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -debug "0,2000" -num_questions 8 -num_episodes_test 200 -policy_path "output/SL_LSTM_32_64/model.pt" -reward "levenshtein" -condition_answer "none" -device_id 1








