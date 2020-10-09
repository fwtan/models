python extract_revisited_features.py \
  --delf_config_path delf_gld_config.pbtxt \
  --dataset_file_path data/gnd_roxford5k.mat \
  --images_dir data/oxford5k/jpg \
  --image_set query \
  --output_features_dir data/oxford5k/delf_features/query


python extract_revisited_features.py \
  --delf_config_path delf_gld_config.pbtxt \
  --dataset_file_path data/gnd_rparis6k.mat \
  --images_dir data/paris6k/jpg \
  --image_set query \
  --output_features_dir data/paris6k/delf_features/query


python extract_revisited_features.py \
  --delf_config_path delf_gld_config.pbtxt \
  --dataset_file_path data/gnd_roxford5k.mat \
  --images_dir data/oxford5k/jpg \
  --image_set index \
  --output_features_dir data/oxford5k/delf_features/index


python extract_revisited_features.py \
  --delf_config_path delf_gld_config.pbtxt \
  --dataset_file_path data/gnd_rparis6k.mat \
  --images_dir data/paris6k/jpg \
  --image_set index \
  --output_features_dir data/paris6k/delf_features/index