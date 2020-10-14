python extract_features.py \
  --delf_config_path r50delg_gldv2clean_config.pbtxt \
  --dataset_file_path data/paris6k/gnd_rparis6k.mat \
  --images_dir data/paris6k/jpg \
  --image_set index \
  --output_features_dir data/paris6k/delg_features_r50/index


python extract_features.py \
  --delf_config_path r101delg_gldv2clean_config.pbtxt \
  --dataset_file_path data/paris6k/gnd_rparis6k.mat \
  --images_dir data/paris6k/jpg \
  --image_set index \
  --output_features_dir data/paris6k/delg_features_r101/index


python extract_features.py \
  --delf_config_path r50delg_gldv2clean_config.pbtxt \
  --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
  --images_dir data/oxford5k/jpg \
  --image_set index \
  --output_features_dir data/oxford5k/delg_features_r50/index


python extract_features.py \
  --delf_config_path r101delg_gldv2clean_config.pbtxt \
  --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
  --images_dir data/oxford5k/jpg \
  --image_set index \
  --output_features_dir data/oxford5k/delg_features_r101/index


python extract_features.py \
  --delf_config_path r50delg_gldv2clean_config.pbtxt \
  --dataset_file_path data/paris6k/gnd_rparis6k.mat \
  --images_dir data/paris6k/jpg \
  --image_set query \
  --output_features_dir data/paris6k/delg_features_r50/query


python extract_features.py \
  --delf_config_path r101delg_gldv2clean_config.pbtxt \
  --dataset_file_path data/paris6k/gnd_rparis6k.mat \
  --images_dir data/paris6k/jpg \
  --image_set query \
  --output_features_dir data/paris6k/delg_features_r101/query


python extract_features.py \
  --delf_config_path r50delg_gldv2clean_config.pbtxt \
  --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
  --images_dir data/oxford5k/jpg \
  --image_set query \
  --output_features_dir data/oxford5k/delg_features_r50/query


python extract_features.py \
  --delf_config_path r101delg_gldv2clean_config.pbtxt \
  --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
  --images_dir data/oxford5k/jpg \
  --image_set query \
  --output_features_dir data/oxford5k/delg_features_r101/query