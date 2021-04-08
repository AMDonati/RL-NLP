#!/bin/bash
#SBATCH --job-name=vqav2data
#SBATCH --qos=qos_gpu-t3
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=vqav2data%j.out
#SBATCH --error=vqav2data%j.err
#SBATCH --time=03:00:00

module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp

export PYTHONPATH=src:${PYTHONPATH}

set -x
srun python -u src/preprocessing/preprocess_vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "none" -min_split 0
srun python -u src/preprocessing/preprocess_vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "data/vqa-v2/cache/vocab.json" -split "train" -min_split 0 -test 1
srun python -u src/preprocessing/preprocess_vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "data/vqa-v2/cache/vocab.json" -split "val" -min_split 0 -test 1
