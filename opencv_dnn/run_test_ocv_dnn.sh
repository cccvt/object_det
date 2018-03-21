#!/usr/bin/env bash
python test_ocv_dnn.py \
--model /home/testuser/PycharmProjects/obj_detector/handtracking/hand_inference_graph/frozen_inference_graph.pb \
--config /home/testuser/PycharmProjects/obj_detector/handtracking/hand_inference_graph/graph.pbtxt \
--framework tensorflow \
--mean 127.5 127.5 127.5 \
--width 300 \
--height 300
