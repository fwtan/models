# python perform_retrieval_rrt.py \
#   --dataset_file_path  data/paris6k/gnd_rparis6k.mat \
#   --query_features_dir data/paris6k/delg_features_r50 \
#   --index_features_dir data/paris6k/delg_features_r50 \
#   --medium_feature_path data/paris6k/paris6k_medium_feats.pkl \
#   --hard_feature_path data/paris6k/paris6k_hard_feats.pkl \
#   --num_local_descriptors 500 \
#   --use_geometric_verification \
#   --output_dir results/paris6k_r50_with_gv_500



python perform_retrieval_rrt.py \
  --dataset_file_path  data/oxford5k/gnd_roxford5k.mat \
  --query_features_dir data/oxford5k/delg_features_r50 \
  --index_features_dir data/oxford5k/delg_features_r50 \
  --medium_feature_path data/oxford5k/oxford5k_medium_feats.pkl \
  --hard_feature_path data/oxford5k/oxford5k_hard_feats.pkl \
  --num_local_descriptors 500 \
  --use_geometric_verification \
  --output_dir results/oxford5k_r50_with_gv_500_rrt