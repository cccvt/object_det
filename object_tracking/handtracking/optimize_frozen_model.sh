#!/usr/bin/env bash
python ~/tensorflow/tensorflow/python/tools/optimize_for_inference.py \
--input /home/testuser/PycharmProjects/obj_detector/test/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb \
--output /home/testuser/PycharmProjects/obj_detector/test/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph_opt.pb \
--input_names image_tensor \
--output_names "num_detections,detection_scores,detection_boxes,detection_classes" \
--placeholder_type_enum 4 \
--frozen_graph