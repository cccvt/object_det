#!/usr/bin/env bash

OBJDET_PATH=/home/testuser/models/research

CFG_PATH=/home/testuser/PycharmProjects/obj_detector/models/model/ssd_mobilenet_v1_pets.config
TRAINDIR=/home/testuser/PycharmProjects/obj_detector/models/model/train
MODELDIR=/home/testuser/PycharmProjects/obj_detector/models/model
EVALDIR=/home/testuser/PycharmProjects/obj_detector/models/model/eval

# run eval job
eval_cmd="python $OBJDET_PATH/object_detection/eval.py \
                --logtostderr \
                --pipeline_config_path=$CFG_PATH \
                --checkpoint_dir=$TRAINDIR \
                --eval_dir=$EVALDIR"

gnome-terminal -e "$eval_cmd" & disown

# run tensorboard monitoring
tensorboard --logdir=$MODELDIR
