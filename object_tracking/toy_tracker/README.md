# add the object_detection forlder to the pythonpath (needed for utils in object_detect.py)
### modify path to point to the object_detection sample folder from the downloaded tensorflow models
### https://github.com/tensorflow/models.git
export PYTHONPATH= ~/models/research/object_detection

# generate csv from all the xmls
python xml_to_csv.py

# split the csv into train and test sets
python split_labels.py

# make tensorflow records from the train and test datasets
python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record
python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record

# run the training job
python ~/models/research/object_detection/train.py \
        --logtostderr \
        --train_dir=models/model/train \
        --pipeline_config_path=models/model/ssd_mobilenet_v1_pets.config

# run the evaluation job
python ~/models/research/object_detection/eval.py \
        --logtostderr \
        --checkpoint_dir=models/model/train \
        --eval_dir=models/model/eval \
        --pipeline_config_path=models/model/ssd_mobilenet_v1_pets.config

# run tensorboard pointing to the folder which contains the train and eval directories
tensorboard --logdir=models/model

# export model to frozen graph
python ~/models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix data/model.ckpt-106 \
    --output_directory object_detection_graph
