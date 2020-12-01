#!/usr/bin/env bash
python src/train/launch_train.py -task "lm" -dataset "vqa" -model "lstm" -num_layers 1 -emb_size 512 -hidden_size 512 -p_drop 0.1 -lr 0.001 -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -out_path "output" -bs 512 -ep 0 -num_workers 6 -model_path "output/vqa_lm_model_correct/vqa_lm_lstm_layers_1_emb_512_hidden_512_pdrop_0.1_gradclip_None_bs_512_lr_0.001/1"
python src/train/launch_train.py -task "policy" -dataset "vqa" -data_path "data/vqa-v2" -out_path "output" -emb_size 512 -hidden_size 1024 -bs 512 -ep 0 -num_workers 6 -fusion "average" -condition_answer "after_fusion" -model_path "output/vqav2_policy_sl/vqa_policy_lstm_layers_1_emb_512_hidden_1024_pdrop_0.0_gradclip_None_bs_512_lr_0.001_cond-answer_after_fusion/1" -features_path "data/vqa-v2/coco_trainval.lmdb"