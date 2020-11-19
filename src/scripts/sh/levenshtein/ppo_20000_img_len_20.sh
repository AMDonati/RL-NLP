#!/bin/bash
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/20000_img_len_20" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 300000 -lm_path "output/lm_model/model.pt" -num_truncated 20 -debug "0,20000" -grad_clip 1 -num_questions 8 -truncate_mode "sample_va" -num_episodes_test 200

python src/scripts/run.py -env "clevr" -reward "lv_norm" -max_len 10 -data_path "data" -out_path "output/RL/CLEVR_exp_Nov" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10000 -lm_path "output/lm_model/model.pt" -debug "0,2000" -grad_clip 1 -num_questions 8 -truncate_mode "proba_thr" -num_episodes_test 100 -p_th 0.05

python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/20000_img_len_20" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 300000 -lm_path "output/lm_model/model.pt" -num_truncated 20 -debug "0,20000" -grad_clip 1 -num_questions 8 -truncate_mode "sample_va" -num_episodes_test 200 -policy_path "output/SL_LSTM_32_64/model.pt"


