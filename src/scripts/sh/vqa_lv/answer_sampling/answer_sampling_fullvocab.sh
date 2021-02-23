#!/usr/bin/env bash
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VILBERT_VQA_fullvocab/temperature_pth_512_1024/answer_sampling/50000_ep" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.01 -lr 0.00005 -word_emb_size 128 -hidden_size 256 -num_episodes_train 100000 -debug "0,20000" -lm_path "output/vqa_lm_model/model.pt" -num_episodes_test 2000 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "average" -condition_answer "after_fusion" -device_id 0 -temperature 1.5 -temp_step 5000 -temp_factor 0.75 -temp_min 0.7 -p_th 0.005 -s_min 5 -s_max 200 -entropy_coeff 0.01 -reward "vilbert_rank2" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -test_metrics "return" "dialog" "bleu" "ppl_dialog_lm" "ttr_question" "sum_probs" "ppl" "lv_norm" "ttr" "selfbleu" "dialogimage" "language_score" "size_valid_actions" -answer_sampl "img_sampling"

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VILBERT_VQA_fullvocab/temperature_pth_512_1024/answer_sampling/50000_ep" -model "lstm" -update_every 32 -agent "PPO" -K_epochs 20 -eps_clip 0.01 -lr 0.00005 -word_emb_size 128 -hidden_size 256 -num_episodes_train 100000 -debug "0,20000" -lm_path "output/vqa_lm_model/model.pt" -num_episodes_test 2000 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "sat" -condition_answer "after_fusion" -device_id 3 -temperature 1.5 -temp_step 5000 -temp_factor 0.75 -temp_min 0.7 -p_th 0.005 -s_min 5 -s_max 200 -entropy_coeff 0.01 -reward "vilbert_rank2" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -test_metrics "return" "dialog" "bleu" "ppl_dialog_lm" "ttr_question" "sum_probs" "ppl" "lv_norm" "ttr" "selfbleu" "dialogimage" "language_score" "size_valid_actions" -answer_sampl "img_sampling"

#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VILBERT_VQA_fullvocab/temperature_pth_512_1024/answer_sampling/50000_ep" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.01 -lr 0.00005 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 100000 -debug "0,2000" -lm_path "output/vqa_lm_model/model.pt" -num_episodes_test 974 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "average" -condition_answer "after_fusion" -device_id 1 -temperature 1.5 -temp_step 5000 -temp_factor 0.75 -temp_min 1 -p_th 0.005 -s_min 1 -s_max 200 -entropy_coeff 0.01 -reward "vilbert_rank2" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -test_metrics "return" "dialog" "bleu" "ppl_dialog_lm" "ttr_question" "sum_probs" "ppl" "lv_norm" "ttr" "selfbleu" "dialogimage" "language_score" "size_valid_actions" -answer_sampl "img_sampling"

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VILBERT_VQA_fullvocab/temperature_pth_512_1024/answer_sampling/50000_ep/baseline_128_256" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.01 -lr 0.00001 -word_emb_size 128 -hidden_size 256 -num_episodes_train 100000 -debug "0,20000" -lm_path "output/vqa_lm_model/model.pt" -reward "vilbert_rank2" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -num_episodes_test 2000 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -device_id 1 -policy_path "output/vqa_policy_128_256_answer/model.pt" -answer_sampl "img_sampling"

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VILBERT_VQA_fullvocab/temperature_pth_512_1024/answer_sampling/50000_ep/baseline_128_256_attn" -model "lstm" -update_every 32 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.01 -lr 0.00001 -word_emb_size 128 -hidden_size 256 -num_episodes_train 100000 -debug "0,20000" -lm_path "output/vqa_lm_model/model.pt" -reward "vilbert_rank2" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -num_episodes_test 2000 -mask_answers 1 -fusion "sat" -condition_answer "after_fusion" -device_id 2 -policy_path "output/vqa_policy_128_256_answer_attn/model.pt" -answer_sampl "img_sampling"


#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VILBERT_VQA_fullvocab/temperature_pth_512_1024/answer_sampling/50000_ep" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.01 -lr 0.00005 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 100000 -debug "0,2000" -lm_path "output/vqa_lm_model/model.pt" -num_episodes_test 5000 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "average" -condition_answer "after_fusion" -device_id 0 -temperature 1.5 -temp_step 5000 -temp_factor 0.75 -temp_min 0.7 -p_th 0.005 -s_min 5 -s_max 200 -entropy_coeff 0.01 -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -test_metrics "return" "dialog" "bleu" "ppl_dialog_lm" "ttr_question" "sum_probs" "ppl" "lv_norm" "ttr" "selfbleu" "dialogimage" "language_score" "size_valid_actions" -answer_sampl "img_sampling"

#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VILBERT_VQA_fullvocab/temperature_pth_512_1024/answer_sampling/50000_ep" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.01 -lr 0.00005 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 100000 -debug "0,2000" -lm_path "output/vqa_lm_model/model.pt" -num_episodes_test 5000 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "average" -condition_answer "after_fusion" -device_id 1 -temperature 1.5 -temp_step 5000 -temp_factor 0.75 -temp_min 1 -p_th 0.005 -s_min 1 -s_max 200 -entropy_coeff 0.01 -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -test_metrics "return" "dialog" "bleu" "ppl_dialog_lm" "ttr_question" "sum_probs" "ppl" "lv_norm" "ttr" "selfbleu" "dialogimage" "language_score" "size_valid_actions" -answer_sampl "img_sampling"

#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VILBERT_VQA_fullvocab/temperature_pth_512_1024/answer_sampling/50000_ep" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.01 -lr 0.00001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 100000 -debug "0,2000" -lm_path "output/vqa_lm_model/model.pt" -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -num_episodes_test 5000 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -device_id 2 -policy_path "output/vqa_policy_128_256_answer/model.pt" -answer_sampl "img_sampling"

python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VILBERT_VQA_fullvocab/temperature_pth_512_1024/answer_sampling/50000_ep/SL_128_256" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.01 -lr 0.00001 -word_emb_size 128 -hidden_size 256 -num_episodes_train 0 -debug "0,2000" -lm_path "output/vqa_lm_model/model.pt" -reward "vilbert_rank2" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -num_episodes_test 5000 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -device_id 1 -policy_path "output/vqa_policy_128_256_answer/model.pt" -answer_sampl "img_sampling"

python src/scripts/test.py -models_path "output/RL/VILBERT_VQA_fullvocab/temperature_pth_512_1024/answer_sampling/50000_ep/vqa_vilbert_rank2_PPO_answ-img_sampling_proba_thr0.005_adam_5e-05_ent0.01_epsclip0.01_graclip5.0_temp1.5_div0.75_step5000_tmin0.7_tmax10.0_smin5_smax200" -num_episodes_test 5000

python src/scripts/test.py -models_path "output/RL/VILBERT_VQA_fullvocab/temperature_pth_512_1024/answer_sampling/50000_ep/baseline_128_256/vqa_vilbert_rank2_REINFORCE_answ-img_sampling_pretrain__adam_1e-05_ent0.01_epsclip0.01_graclipNone" -num_episodes_test 974

