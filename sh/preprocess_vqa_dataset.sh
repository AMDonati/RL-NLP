#!/usr/bin/env bash
echo "------- creating full vocab --------------------"
python src/preprocessing/preprocess_vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "none" -min_split 0
echo "------- creating reduced vocab --------------------"
python src/preprocessing/preprocess_vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "none" -min_split 1
echo "-----------creating train_entries.pkl file ----------"
python src/preprocessing/preprocess_vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "data/vqa-v2/cache/vocab.json" -split "train" -min_split 0 -test 1
echo "-----------creating val_entries.pkl file ----------"
python src/preprocessing/preprocess_vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "data/vqa-v2/cache/vocab.json" -split "val" -min_split 0 -test 1
echo "-----------creating mintrain_entries.pkl file ----------"
python src/preprocessing/preprocess_vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "data/vqa-v2/cache/vocab.json" -split "train" -min_split 1 -test 0
echo "-----------creating minval_entries.pkl file ----------"
python src/preprocessing/preprocess_vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "data/vqa-v2/cache/vocab.json" -split "val" -min_split 1 -test 0
echo "-----------creating mintrain_minvocab_entries.pkl file ----------"
python src/preprocessing/preprocess_vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "data/vqa-v2/cache/vocab_min.json" -split "train" -min_split 1 -test 0
echo "-----------creating minval_minvocab_entries.pkl file ----------"
python src/preprocessing/preprocess_vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "data/vqa-v2/cache/vocab_min.json" -split "val" -min_split 1 -test 0



