python perform_retrieval.py \
  --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
  --query_features_dir data/oxford5k/delg_features_r50 \
  --index_features_dir data/oxford5k/delg_features_r50 \
  --use_geometric_verification \
  --output_dir results/oxford5k_r50_with_gv

python perform_retrieval.py \
  --dataset_file_path  data/paris6k/gnd_rparis6k.mat \
  --query_features_dir data/paris6k/delg_features_r50 \
  --index_features_dir data/paris6k/delg_features_r50 \
  --use_geometric_verification \
  --output_dir results/paris6k_r50_with_gv

# python perform_retrieval.py \
#   --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
#   --query_features_dir data/oxford5k/delg_features_r101/query \
#   --index_features_dir data/oxford5k/delg_features_r101/index \
#   --use_geometric_verification \
#   --output_dir results/oxford5k_r101_with_gv

# python perform_retrieval.py \
#   --dataset_file_path  data/paris6k/gnd_rparis6k.mat \
#   --query_features_dir data/paris6k/delg_features_r101/query \
#   --index_features_dir data/paris6k/delg_features_r101/index \
#   --use_geometric_verification \
#   --output_dir results/paris6k_r101_with_gv