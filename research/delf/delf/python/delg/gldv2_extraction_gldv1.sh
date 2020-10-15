python extract_gldv2_features.py \
  --delf_config_path r50delg_gld_config.pbtxt \
  --dataset_file_path data/gldv2/train_local.txt \
  --images_dir data/gldv2/ \
  --output_features_dir data/gldv2/delg_features

python extract_gldv2_features.py \
  --delf_config_path r50delg_gld_config.pbtxt \
  --dataset_file_path data/gldv2/test_query.txt \
  --images_dir data/gldv2/ \
  --output_features_dir data/gldv2/delg_features_query

python extract_gldv2_features.py \
  --delf_config_path r50delg_gld_config.pbtxt \
  --dataset_file_path data/gldv2/test_gallery.txt \
  --images_dir data/gldv2/ \
  --output_features_dir data/gldv2/delg_features_index