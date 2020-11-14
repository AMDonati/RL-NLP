#!/usr/bin/env bash
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt-2" -reward "levenshtein" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 1 -fusion "average" -init_text "The question is:" -custom_init 0

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt-2" -reward "levenshtein" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 1 -fusion "average" -init_text "Ask me a question." -custom_init 0

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt-2" -reward "levenshtein" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 1 -fusion "average" -init_text "I would like to ask you a question." -custom_init 0


python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt-2" -reward "levenshtein" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 1 -fusion "average" -init_text "Here are a few examples:" -custom_init 10

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt-2" -reward "levenshtein" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 1 -fusion "average" -init_text "Here are a few examples:" -custom_init 100

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt-2" -reward "levenshtein" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 1 -fusion "average" -init_text "Here are a few examples:" -custom_init 500

