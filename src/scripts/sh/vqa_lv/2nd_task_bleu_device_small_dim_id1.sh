#!/usr/bin/env bash
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,10" -lm_path "output/vqa_lm_model/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -p_th 0.01 -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 1

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,10" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -p_th 0.01 -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 1

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -p_th 0.01 -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 1

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -p_th 0.01 -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 1

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,10" -lm_path "output/vqa_lm_model/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -p_th 0.005 -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 1

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,10" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -p_th 0.005 -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 1

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -p_th 0.005 -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 1

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -p_th 0.005 -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 1

#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/temp" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 16 -hidden_size 16 -num_episodes_train 5 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 20 -mask_answers 1 -truncate_mode "proba_thr" -p_th 0.005 -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2







