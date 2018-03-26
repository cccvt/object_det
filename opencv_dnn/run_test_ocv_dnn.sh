#!/usr/bin/env bash
python test_ocv_dnn.py \
--model /home/testuser/obj_det_git/object_tracking/handtracking2/hand_inference_graph/frozen_inference_graph.pb \
--config /home/testuser/obj_det_git/object_tracking/handtracking2/hand_inference_graph/fixed_graph.pbtxt \
--framework tensorflow \
--mean 127.5 127.5 127.5 \
--width 300 \
--height 300
