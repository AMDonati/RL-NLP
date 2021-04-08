#!/bin/bash
#SBATCH --job-name=clevrfeatures
#SBATCH --qos=qos_gpu-t3
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=clevrfeatures%j.out
#SBATCH --error=clevrfeatures%j.err
#SBATCH --time=03:00:00

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp

export PYTHONPATH=src:${PYTHONPATH}

set -x
srun python -u src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_train_questions.json" -out_vocab_path "data/vocab.json" -out_h5_path "data/train_questions.h5" -min_token_count 1
srun python -u src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_val_questions.json" -out_vocab_path "data/vocab.json" -out_h5_path "data/val_questions.h5" -min_token_count 1
srun python -u src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_test_questions.json" -out_vocab_path "data/vocab.json" -out_h5_path "data/test_questions.h5" -min_token_count 1
srun python -u src/preprocessing/extract_features.py --input_image_dir data/CLEVR_v1.0/images/train --output_h5_file data/train_features.h5 --batch_size 64
srun python -u src/preprocessing/extract_features.py --input_image_dir data/CLEVR_v1.0/images/val --output_h5_file data/val_features.h5 --batch_size 64
srun python -u src/preprocessing/extract_features.py --input_image_dir data/CLEVR_v1.0/images/test --output_h5_file data/test_features.h5 --batch_size 64
