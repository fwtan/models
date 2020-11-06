# python perform_retrieval_d1m.py \
#   --dataset_file_path data/revisitop1m/gnd_roxford5k.mat \
#   --query_features_dir data/revisitop1m/delg_features_r50 \
#   --index_features_dir data/revisitop1m/delg_features_r50 \
#   --d1m_file_path data/revisitop1m/revisitop1m.txt \
#   --use_geometric_verification \
#   --output_dir results/d1m_oxford5k_r50_with_gv

python perform_retrieval_d1m.py \
  --dataset_file_path  data/revisitop1m/gnd_rparis6k.mat \
  --query_features_dir data/revisitop1m/delg_features_r50 \
  --index_features_dir data/revisitop1m/delg_features_r50 \
  --d1m_file_path data/revisitop1m/revisitop1m.txt \
  --use_geometric_verification \
  --output_dir results/d1m_paris6k_r50_with_gv