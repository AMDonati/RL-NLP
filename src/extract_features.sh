python src/preprocessing/extract_features.py \
  --input_image_dir data/CLEVR_v1.0/images/train \
  --output_h5_file data/train_features.h5 --batch_size 64

python src/preprocessing/scripts/extract_features.py \
  --input_image_dir data/CLEVR_v1.0/images/val \
  --output_h5_file data/val_features.h5 --batch_size 64

python src/preprocessing/extract_features.py \
  --input_image_dir data/CLEVR_v1.0/images/test \
  --output_h5_file data/test_features.h5 --batch_size 64