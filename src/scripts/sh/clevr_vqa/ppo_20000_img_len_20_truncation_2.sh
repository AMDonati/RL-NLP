#!/bin/bash
echo "-------------------------- proba thr 0.05---------------------------------------------------------------------------------------------------"
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2021_exp_clvr_vqa_20000img_len20" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 5000 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 1 -device_id 2 -temperature 1.5 -temp_step 1000 -temp_factor 0.75 -temp_min 0.7 -s_min 5
echo "-------------------------- proba thr 1/V---------------------------------------------------------------------------------------------------"
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2021_exp_clvr_vqa_20000img_len20" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 5000 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -truncate_mode "proba_thr" -p_th 0.05 -grad_clip 1 -device_id 2 -temperature 1.5 -temp_step 1000 -temp_factor 0.75 -temp_min 0.7 -s_min 5



