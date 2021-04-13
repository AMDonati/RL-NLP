#!/bin/bash
#SBATCH --job-name=vqav2data-pp
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=vqav2data-pp%j.out
#SBATCH --error=vqav2data-pp%j.err

module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp

export PYTHONPATH=src:${PYTHONPATH}

set -x
srun python -u src/preprocessing/preprocess_vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "none" -min_split 1
srun python -u src/preprocessing/preprocess_vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "data/vqa-v2/cache/vocab_min.json" -split "train" -min_split 1 -test 0
srun python -u src/preprocessing/preprocess_vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "data/vqa-v2/cache/vocab_min.json" -split "val" -min_split 1 -test 0
#srun python -u src/preprocessing/preprocess_vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "none" -min_split 0
#srun python -u src/preprocessing/preprocess_vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "data/vqa-v2/cache/vocab.json" -split "train" -min_split 0 -test 1
#srun python -u src/preprocessing/preprocess_vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "data/vqa-v2/cache/vocab.json" -split "val" -min_split 0 -test 1
