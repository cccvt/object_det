#!/usr/bin/env bash

OBJDET_PATH=/home/testuser/models/research

CFG_PATH=/home/testuser/PycharmProjects/obj_detector/models/model/ssd_mobilenet_v1_pets.config
TRAINDIR=/home/testuser/PycharmProjects/obj_detector/models/model/train
MODELDIR=/home/testuser/PycharmProjects/obj_detector/models/model
EVALDIR=/home/testuser/PycharmProjects/obj_detector/models/model/eval
FREEZEDIR=/home/testuser/PycharmProjects/obj_detector/models/frozen_models

# run export job
freeze_cmd="python $OBJDET_PATH/object_detection/export_inference_graph.py \
                --input_type=image_tensor \
                --pipeline_config_path=$CFG_PATH \
                --trained_checkpoint_prefix=$TRAINDIR/model.ckpt-1275 \
                --output_directory=$FREEZEDIR"

gnome-terminal -e "$freeze_cmd" & disown