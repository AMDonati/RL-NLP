#!/usr/bin/env bash
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_smallvocab_vilbert/temperature_pth_512_1024/50000_ep" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.01 -lr 0.00005 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 50000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 974 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "average" -condition_answer "after_fusion" -device_id 0 -min_data 1 -temperature 2 -temp_step 1000 -temp_factor 0.75 -temp_min 0.7 -p_th 0.01 -s_min 1 -s_max 200 -entropy_coeff 0.01 -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -test_metrics "return" "dialog" "bleu" "ppl_dialog_lm" "ttr_question" "sum_probs" "ppl" "lv_norm" "ttr" "selfbleu" "dialogimage" "language_score" "size_valid_actions"






#TODO: to relaunch.
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_smallvocab_vilbert/temperature_pth_512_1024/50000_ep" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.01 -lr 0.00005 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 50000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "average" -condition_answer "after_fusion" -device_id 1 -min_data 1 -temperature 2 -temp_step 1000 -temp_factor 0.75 -temp_min 0.7 -p_th 0.005 -s_min 1 -s_max 200 -entropy_coeff 0.015 -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin"


python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_smallvocab_vilbert/temperature_pth_512_1024/50000_ep/s_min10" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.01 -lr 0.00005 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 50000 -debug "0,2000" -lm_path "gpt" -num_episodes_test 974 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "average" -condition_answer "after_fusion" -device_id 0 -min_data 1 -temperature 2 -temp_step 1000 -temp_factor 0.75 -temp_min 0.7 -p_th 0.005 -s_min 10 -s_max 200 -entropy_coeff 0.01 -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -init_text "Here are a few examples:" -custom_init 50 -test_metrics "return" "dialog" "bleu" "ppl_dialog_lm" "ttr_question" "sum_probs" "ppl" "lv_norm" "ttr" "selfbleu" "dialogimage" "language_score"

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_smallvocab_vilbert/temperature_pth_512_1024/50000_ep/s_min10" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.01 -lr 0.00005 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 50000 -debug "0,2000" -lm_path "gpt" -num_episodes_test 974 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "average" -condition_answer "after_fusion" -device_id 1 -min_data 1 -temperature 2 -temp_step 1000 -temp_factor 0.75 -temp_min 0.7 -p_th 0.01 -s_min 10 -s_max 200 -entropy_coeff 0.01 -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -init_text "Here are a few examples:" -custom_init 50 -test_metrics "return" "dialog" "bleu" "ppl_dialog_lm" "ttr_question" "sum_probs" "ppl" "lv_norm" "ttr" "selfbleu" "dialogimage" "language_score"

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_smallvocab_vilbert/temperature_pth_512_1024/50000_ep/s_min10" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.01 -lr 0.00005 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 50000 -debug "0,2000" -lm_path "gpt" -num_episodes_test 974 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "average" -condition_answer "after_fusion" -device_id 2 -min_data 1 -temperature 2 -temp_step 1000 -temp_factor 0.75 -temp_min 0.7 -p_th 0.003 -s_min 10 -s_max 200 -entropy_coeff 0.01 -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -init_text "Here are a few examples:" -custom_init 50 -test_metrics "return" "dialog" "bleu" "ppl_dialog_lm" "ttr_question" "sum_probs" "ppl" "lv_norm" "ttr" "selfbleu" "dialogimage" "language_score"







python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_smallvocab_vilbert/temperature_pth_512_1024/50000_ep" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.01 -lr 0.00001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 50000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -device_id 3 -min_data 1 -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt"

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_smallvocab_vilbert/temperature_pth_512_1024/50000_ep/SL_baseline" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.01 -lr 0.00001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 0 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -num_episodes_test 974 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -device_id 3 -min_data 1 -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -test_metrics "return" "dialog" "bleu" "ppl_dialog_lm" "ttr_question" "sum_probs" "ppl" "lv_norm" "ttr" "selfbleu" "dialogimage" "language_score"



#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_smallvocab_vilbert/scratch" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.02 -lr 0.00001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 20000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -device_id 3 -min_data 1

#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_smallvocab_vilbert/scratch" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.02 -lr 0.00001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 20000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -device_id 3 -min_data 1

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_smallvocab_vilbert/temperature_pth_512_1024/50000_ep/pretrain" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.01 -lr 0.00001 -optimizer "sgd" -word_emb_size 512 -hidden_size 1024 -num_episodes_train 50000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -num_episodes_test 974 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -truncate_mode "proba_thr" -p_th 0.005 -device_id 0 -min_data 1 -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -test_metrics "return" "dialog" "bleu" "ppl_dialog_lm" "ttr_question" "sum_probs" "ppl" "lv_norm" "ttr" "selfbleu" "dialogimage" "language_score"

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_smallvocab_vilbert/temperature_pth_512_1024/50000_ep/pretrain" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.01 -lr 0.000001 -optimizer "sgd" -word_emb_size 512 -hidden_size 1024 -num_episodes_train 50000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -num_episodes_test 974 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -truncate_mode "proba_thr" -p_th 0.005 -device_id 1 -min_data 1 -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -test_metrics "return" "dialog" "bleu" "ppl_dialog_lm" "ttr_question" "sum_probs" "ppl" "lv_norm" "ttr" "selfbleu" "dialogimage" "language_score"

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_smallvocab_vilbert/temperature_pth_512_1024/50000_ep/pretrain" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.01 -lr 0.00001 -optimizer "sgd" -word_emb_size 512 -hidden_size 1024 -num_episodes_train 50000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -num_episodes_test 974 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -truncate_mode "proba_thr" -p_th 0.001 -device_id 2 -min_data 1 -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -test_metrics "return" "dialog" "bleu" "ppl_dialog_lm" "ttr_question" "sum_probs" "ppl" "lv_norm" "ttr" "selfbleu" "dialogimage" "language_score"

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_smallvocab_vilbert/temperature_pth_512_1024/50000_ep/pretrain" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.01 -lr 0.00001 -optimizer "sgd" -word_emb_size 512 -hidden_size 1024 -num_episodes_train 50000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -num_episodes_test 974 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -truncate_mode "proba_thr" -p_th 0.005 -temperature 2 -temp_step 1000 -temp_factor 0.75 -temp_min 0.7 -device_id 3 -min_data 1 -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -test_metrics "return" "dialog" "bleu" "ppl_dialog_lm" "ttr_question" "sum_probs" "ppl" "lv_norm" "ttr" "selfbleu" "dialogimage" "language_score"