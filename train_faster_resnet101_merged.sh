# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH=dog_detector/models/faster_rcnn_resnet101_merged/pipeline.config
MODEL_DIR=dog_detector/models/faster_rcnn_resnet101_merged
NUM_TRAIN_STEPS=200000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python /usr/local/lib/python3.5/dist-packages/tensorflow/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
