# From models/research/delf/delf/python/detect_to_retrieve
# python cluster_delg_features.py \
#   --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
#   --features_dir data/oxford5k/delg_features_r50/index \
#   --num_clusters 1024 \
#   --num_iterations 50 \
#   --output_cluster_dir data/oxford5k_clusters_1024


# python cluster_delg_features.py \
#   --dataset_file_path data/paris6k/gnd_rparis6k.mat \
#   --features_dir data/paris6k/delg_features_r50/index \
#   --num_clusters 65536 \
#   --num_iterations 50 \
#   --output_cluster_dir data/paris6k_clusters_65536


python cluster_delg_features.py \
  --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
  --features_dir data/oxford5k/delg_features_r50/index \
  --num_clusters 65536 \
  --num_iterations 50 \
  --output_cluster_dir data/oxford5k_clusters_65536