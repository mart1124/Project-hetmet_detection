import cv2
import numpy as np
import imutils
import time
from threading import Thread
from os import path
from datetime import datetime
from numpy.core.numeric import rollaxis
from yolo import YOLOv4
from centroidtracker import CentroidTracker
 
import tf_objectdetection as tf_ob
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


#### H=1080,W=1920 Video Path ####
Camera_Path = "rtsp://admin:Total9999%2B@192.168.1.50/Streaming/Channels/1"
Video_Path = "./data/video/resize.mp4"
#### Yolo Path ####
data_path = "./data"
weightsPath = "./weights/yolov4-custom_final.weights"
configPath = data_path + "/" + "yolov4-custom.cfg"
labelsPath = data_path + "/labels/" + "objv4.names"
#### Tensorflow Path ####
tf.saved_model.load("./data/data_tensorflow/motorbike_model/exporter/saved_model")
tf.saved_model.load("./data/data_tensorflow/person_model/person_export/saved_model")
category_index = label_map_util.create_category_index_from_labelmap(data_path+ '/' + 'labels/motorbike_label_map.txt')
category_index_model2 = label_map_util.create_category_index_from_labelmap(data_path+ '/' +'labels/person_label_map.txt')
print(category_index,category_index_model2)
#### output_Path
BASE_DIR = './data/output'
PREFIX = 'Object-Detection'
PREFIX_JPG = "Image"
EXTENSION = ['avi','jpg']
file_name_format = "{:s}-{:d}-{:%d_%m_%Y_%H%M%S}.{:s}"
# print(file_name_format)
date = datetime.now()
not_detected_posture = 0
file_name = file_name_format.format(PREFIX, not_detected_posture, date, EXTENSION[0])
file_path = path.normpath(path.join(BASE_DIR, file_name))
img_file_name = file_name_format.format(PREFIX_JPG, not_detected_posture, date, EXTENSION[1])
img_file_path = path.normpath(path.join(BASE_DIR, img_file_name))
frame_count = 0
# print(file_path)

#### Check Video Error ####
try:
    cap = cv2.VideoCapture(Video_Path)
    if not cap.isOpened():
        raise NameError('Please select the correct video file.')
except cv2.error as e:
    print("cv2.error:", e)
except Exception as e:
    print("Exception:", e)
else:
    print("Open Video file...")
    time.sleep(0.5)
    print("Start Program")
    time.sleep(0.5)

codec = cv2.VideoWriter_fourcc(*'XVID')
cap_fps =int(cap.get(cv2.CAP_PROP_FPS))
cap_width,cap_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(file_path, codec, cap_fps, (cap_width, cap_height))
# ret, frame2 = cap.read()

#### ตัวแปรต่างๆ ####
cuda = True
(H,W) = (None,None)
starting_time = time.time()
frame_id = 0 
#### yolo ####

ct = CentroidTracker(maxDisappeared=10, maxDistance=80)
model = YOLOv4(weightsPath, configPath, labelsPath, confidence_threshold=0.5, nms_threshold=0.6)

def motorcycle_detect(frame, frame_count, category_index, category_index_model2):
    image_np = np.array(frame)
    frame_count += 1
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = tf_ob.detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    output_var, boxx = viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=4,
                min_score_thresh=.8,
                agnostic_mode=False)
    
    im_height, im_width = frame.shape[:2]
    rects = []
    if len(boxx) > 0:
        for item in boxx:
            ymin, xmin, ymax, xmax = item
            (left, right, bottom, top) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
            rects.append([int(left),int(top),int(right),int(bottom)])
            print(rects)
            center_x = int((xmin + xmax)*im_width/2)
            # print(center_x)
                # roi = frame[int(top)-70:int(bottom)+70, int(left)-30:int(right)+70] ## จับวัตถุที่สนใจ
            roi = frame[int(bottom)-70:int(top)+70, int(left)-20:int(right)+70] ## จับวัตถุที่สนใจ
                # detect_roi = motorcycle_detect_roi(roi,category_index_model2,label_id_offset)
            if center_x <= (int(3*im_width/6+im_width/50)) and center_x >= (int(3*im_width/6-im_width/50)):
                threadProcessImage = Thread(target = model.detect(roi))
                threadProcessImage.start()
            # out.write(image_np_with_detections)
            cv2.imshow("Roi_objecct", roi)
    objects = ct.update(rects)
    if objects != None:
        # for (objectID, centroid) in objects.items():
        for (objectID, boxx) in objects.items():  #เพิม่
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            x1, y1, x2, y2 = boxx
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            
            text = "ID {}".format(objectID)
            # cv2.putText(image_np_with_detections, text, (centroid[0] - 10, centroid[1] - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image_np_with_detections, text, (cX - 10, cY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(image_np_with_detections, (cX, cY), 4, (0, 255, 0), -1)
            

    cv2.line(image_np_with_detections, (int(3*im_width/6+im_width/50),0), (int(3*im_width/6-im_width/50),1080),(255,0,0),2)
    cv2.line(image_np_with_detections, (int(3*im_width/6+im_width/50),0), (int(3*im_width/6+im_width/50),1080), (0, 255, 0), thickness=2)
    cv2.line(image_np_with_detections, (int(3*im_width/6-im_width/50),0), (int(3*im_width/6-im_width/50),1080), (0, 255, 0), thickness=2)
        
    return image_np_with_detections, rects

def motorcycle_detect_roi(frame, category_index_model2, label_id_offset) :
    image_np_roi = np.array(frame)

    input_tensor_roi = tf.convert_to_tensor(np.expand_dims(image_np_roi, 0), dtype=tf.float32)
    detections_model2 = tf_ob.detect_fn2(input_tensor_roi)

    num_detections_model2 = int(detections_model2.pop('num_detections'))
    detections_model2 = {key: value[0, :num_detections_model2].numpy()
                for key, value in detections_model2.items()}
    detections_model2['num_detections'] = num_detections_model2

    # detection_classes should be ints.
    detections_model2['detection_classes'] = detections_model2['detection_classes'].astype(np.int64)

    image_np_with_detections_2 = image_np_roi.copy()

    output_var, boxx = viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections_2,
        detections_model2['detection_boxes'],
        detections_model2['detection_classes']+label_id_offset,
        detections_model2['detection_scores'],
        category_index_model2,
        use_normalized_coordinates=True,
        max_boxes_to_draw=2,
        min_score_thresh=.3,
        agnostic_mode=False)

    return image_np_with_detections_2


while True:
    ret, frame1 = cap.read()
    if frame1 is None:
        print('Completed')
        break
    frame_id += 1
    frame_resize1 = imutils.resize(frame1, width=980) #H=551 ,W=980
    if H is None or W is None:
        (H,W) = frame_resize1.shape[:2]

    result, rects = motorcycle_detect(frame_resize1, frame_count, category_index, category_index_model2)
    
    #### แสดงผลการตรวจจับ
    # result = imutils.resize(result, width=980)
    cv2.imshow("Object Detection", result)
    
    # frame1 = frame2
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
