python perform_retrieval_ablation.py \
  --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
  --query_features_dir data/oxford5k/delg_features_r50 \
  --index_features_dir data/oxford5k/delg_features_r50 \
  --num_local_descriptors 200 \
  --use_geometric_verification \
  --output_dir results/oxford5k_r50_with_gv_200


python perform_retrieval_ablation.py \
  --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
  --query_features_dir data/oxford5k/delg_features_r50 \
  --index_features_dir data/oxford5k/delg_features_r50 \
  --num_local_descriptors 400 \
  --use_geometric_verification \
  --output_dir results/oxford5k_r50_with_gv_400


python perform_retrieval_ablation.py \
  --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
  --query_features_dir data/oxford5k/delg_features_r50 \
  --index_features_dir data/oxford5k/delg_features_r50 \
  --num_local_descriptors 500 \
  --use_geometric_verification \
  --output_dir results/oxford5k_r50_with_gv_500


python perform_retrieval_ablation.py \
  --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
  --query_features_dir data/oxford5k/delg_features_r50 \
  --index_features_dir data/oxford5k/delg_features_r50 \
  --num_local_descriptors 600 \
  --use_geometric_verification \
  --output_dir results/oxford5k_r50_with_gv_600


python perform_retrieval_ablation.py \
  --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
  --query_features_dir data/oxford5k/delg_features_r50 \
  --index_features_dir data/oxford5k/delg_features_r50 \
  --num_local_descriptors 800 \
  --use_geometric_verification \
  --output_dir results/oxford5k_r50_with_gv_800