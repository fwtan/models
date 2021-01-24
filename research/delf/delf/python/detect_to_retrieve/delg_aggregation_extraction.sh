# From models/research/delf/delf/python/detect_to_retrieve
# python extract_aggregation_delg.py \
#   --use_query_images True \
#   --aggregation_config_path query_aggregation_config_delg.pbtxt \
#   --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
#   --features_dir data/oxford5k/delg_features_r50/query \
#   --output_aggregation_dir data/oxford5k_aggregation/query_delg


python extract_aggregation_delg.py \
  --use_query_images True \
  --aggregation_config_path query_aggregation_config_delg.pbtxt \
  --dataset_file_path data/paris6k/gnd_rparis6k.mat \
  --features_dir data/paris6k/delg_features_r50 \
  --output_aggregation_dir data/paris6k_aggregation/query_delg 


# python extract_aggregation_delg.py \
#   --aggregation_config_path index_aggregation_config_delg.pbtxt \
#   --dataset_file_path data/paris6k/gnd_rparis6k.mat \
#   --features_dir data/paris6k/delg_features_r50 \
#   --output_aggregation_dir data/paris6k_aggregation/index_delg 