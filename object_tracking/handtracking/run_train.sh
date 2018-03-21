#!/usr/bin/env bash

OBJDET_PATH=/home/testuser/models/research

CFG_PATH=/home/testuser/PycharmProjects/obj_detector/models/model/ssd_mobilenet_v1_pets.config
TRAINDIR=/home/testuser/PycharmProjects/obj_detector/models/model/train
MODELDIR=/home/testuser/PycharmProjects/obj_detector/models/model
EVALDIR=/home/testuser/PycharmProjects/obj_detector/models/model/eval

# run training job
train_cmd="python $OBJDET_PATH/object_detection/train.py \
                --logtostderr \
                --pipeline_config_path=$CFG_PATH \
                --train_dir=$TRAINDIR"

gnome-terminal -e "$train_cmd" & disown
