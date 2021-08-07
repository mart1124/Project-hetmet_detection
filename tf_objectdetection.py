import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import time

CHECKPOINT_PATH = "./data/data_tensorflow/motorbike_model/"
CONFIG_PATH = CHECKPOINT_PATH + "motorbike.config"
PATH_TO_MODEL_DIR = CHECKPOINT_PATH + "exporter"
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

CHECKPOINT_PATH_Model2 = "./data/data_tensorflow/person_model/"
CONFIG_PATH_Model2 = CHECKPOINT_PATH_Model2 + "person.config"
PATH_TO_MODEL_DIR_Model2 = CHECKPOINT_PATH_Model2 + "person_export"
PATH_TO_SAVED_MODEL_Model2 = PATH_TO_MODEL_DIR_Model2 + "/saved_model"
# Load pipeline config and build a detection model
print('Loading model...', end='')
start_time = time.time()

#### Model-1
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-26')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

#### Model-2
configs_model2 = config_util.get_configs_from_pipeline_file(CONFIG_PATH_Model2)
detection_model2 = model_builder.build(model_config=configs_model2['model'], is_training=False)

# Restore checkpoint
ckpt_2 = tf.compat.v2.train.Checkpoint(model=detection_model2)
ckpt_2.restore(os.path.join(CHECKPOINT_PATH_Model2, 'ckpt-6')).expect_partial()

def detect_fn2(image):
    image, shapes = detection_model2.preprocess(image)
    prediction_dict = detection_model2.predict(image, shapes)
    detections2 = detection_model2.postprocess(prediction_dict, shapes)
    return detections2


end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))