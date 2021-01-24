# From models/research/delf/delf/python/detect_to_retrieve
python perform_retrieval_delg_global.py \
  --index_aggregation_config_path index_aggregation_config_delg.pbtxt \
  --query_aggregation_config_path query_aggregation_config_delg.pbtxt \
  --dataset_file_path  data/paris6k/gnd_rparis6k.mat \
  --index_aggregation_dir data/paris6k_aggregation/index_delg \
  --query_aggregation_dir data/paris6k_aggregation/query_delg \
  --output_dir results/paris6k_aggregation_global