python perform_retrieval.py \
  --dataset_file_path data/revisitop1m/gnd_roxford5k.mat \
  --query_features_dir data/revisitop1m/delg_features_r50 \
  --index_features_dir data/revisitop1m/delg_features_r50 \
  --use_geometric_verification \
  --output_dir results/oxford5k_r50_with_gv

python perform_retrieval.py \
  --dataset_file_path  data/revisitop1m/gnd_rparis6k.mat \
  --query_features_dir data/revisitop1m/delg_features_r50 \
  --index_features_dir data/revisitop1m/delg_features_r50 \
  --use_geometric_verification \
  --output_dir results/paris6k_r50_with_gv
