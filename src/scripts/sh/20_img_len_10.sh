#!/usr/bin/env bash
python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/20_img_len_10" -model "lstm" -update_every 20 -agent "PPO" -K_epochs 10 -eps_clip 0.02 -word_emb_size 8 -hidden_size 24 -num_episodes_train 30000 -lm_path "output/lstm_layers_1_emb_32_hidden_64_pdrop_0.0_gradclip_None_bs_512_lr_0.001/model.pt" -num_truncated 10 -debug "0,20" -grad_clip 1 -num_questions 1

python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/20_img_len_10" -model "lstm" -update_every 20 -agent "PPO" -K_epochs 10 -eps_clip 0.02 -word_emb_size 8 -hidden_size 24 -num_episodes_train 30000 -lm_path "output/lstm_layers_1_emb_32_hidden_64_pdrop_0.0_gradclip_None_bs_512_lr_0.001/model.pt" -num_truncated 30 -debug "0,20" -grad_clip 1 -num_questions 1

python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/20_img_len_10" -model "lstm" -update_every 20 -agent "PPO" -K_epochs 10 -eps_clip 0.02 -word_emb_size 8 -hidden_size 24 -num_episodes_train 30000 -lm_path "output/lstm_layers_1_emb_32_hidden_64_pdrop_0.0_gradclip_None_bs_512_lr_0.001/model.pt" -num_truncated 50 -debug "0,20" -grad_clip 1 -num_questions 1

python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/20_img_len_10" -model "lstm" -update_every 20 -agent "PPO" -K_epochs 10 -eps_clip 0.02 -word_emb_size 8 -hidden_size 24 -num_episodes_train 30000 -lm_path "output/lstm_layers_1_emb_32_hidden_64_pdrop_0.0_gradclip_None_bs_512_lr_0.001/model.pt" -num_truncated 87 -debug "0,20" -grad_clip 1 -num_questions 1
