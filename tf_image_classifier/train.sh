python retrain.py \
  --bottleneck_dir=tf_files/bottlenecks \
  --testing_percentage 20 \
  --how_many_training_steps=500 \
  --model_dir=inception \
  --summaries_dir=tf_files/training_summaries/basic \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --image_dir=/home/testuser/obj_det_git/hand_pose_dataset \
  --architecture mobilenet_0.25_128
