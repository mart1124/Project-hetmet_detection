import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import time

CONFIG_PATH = "./data/data_tensorflow/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.config"
CHECKPOINT_PATH = './data/data_tensorflow/'
PATH_TO_MODEL_DIR = CHECKPOINT_PATH + "/export_model"
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
# Load pipeline config and build a detection model
print('Loading model...', end='')
start_time = time.time()

configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))