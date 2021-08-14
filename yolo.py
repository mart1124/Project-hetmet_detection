#ใช้สำหรับเรียกใช้ Yolov4
import cv2
import numpy as np
from threading import Thread

class YOLOv4():

    def __init__(self, weightsPath, configPath, labelsPath, confidence_threshold=0.5, nms_threshold=0.6):
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        self.object_names = open(labelsPath).read().strip().split("\n")
        ln = self.net.getLayerNames()
        self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.colors = [[0,255,0],[0,0,255]]
    
        # self.img_height = None
        # self.img_width = None

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        # Counter var
        self.counter_helmet = 0
        self.counter_no_helmet = 0

    def forward(self, frame):
        blob =  cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=True)
        self.net.setInput(blob)
        layerOutputs =  self.net.forward(self.ln)
        return layerOutputs

    def detect(self, frame , frame_result):
        # print("/////////in////////")
        # if self.img_height is None or self.img_width is None:
        (self.img_height, self.img_width) = frame.shape[:2]
        layerOutputs = self.forward(frame)

        boxes = []
        confidences = []
        classIDs = []

        # do other cv2 stuff....
        for output in layerOutputs:
        # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.5: #0.5 ค่าเริ่มต้น
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([self.img_width, self.img_height, self.img_width, self.img_height])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.6)
            if len(idxs) > 0:
            # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    # draw a bounding box rectangle and label on the frame
                    color = [int(c) for c in self.colors[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(self.object_names[classIDs[i]],
                        confidences[i])
                    cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        # check if the video writer is Non
                    if self.object_names[classIDs[i]] != None and len(idxs) != None:
                        if self.object_names[classIDs[i]] == "Helmet":
                            self.counter_helmet += 1
                        if self.object_names[classIDs[i]] == "No_Helmet":
                            self.counter_no_helmet += 1
                        return (self.counter_helmet,self.counter_no_helmet)
                    else:
                        return None   
            

