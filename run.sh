python main.py --base_path 'data/Cancerous cell smears' \
  --output_path 'output' \
  --image_type 'BMP' \
  --batch_size 10 \
  --edge_filter_operator '-3 0 3 -10 0 10 -3 0 3' \
  --edge_filter_size '3 3' \
  --k_means_clusters 2 \
  --color_channel 'R'
