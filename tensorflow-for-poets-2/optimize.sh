IMAGE_SIZE=224
toco \
  --input_file=tf_files/retrained_graph.pb \
  --output_file=tf_files/optimized_graph.pb \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TENSORFLOW_GRAPHDEF \
  --input_shape=1,${IMAGE_SIZE},${IMAGE_SIZE},3 \
  --input_array=input \
  --output_array=final_result
