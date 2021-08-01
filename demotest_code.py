import cv2
import numpy as np
import imutils
import time
from threading import Thread

from numpy.core.numeric import rollaxis
from yolo import YOLOv4


#### H=1080,W=1920 Video Path ####
Camera_Path = "rtsp://admin:Total9999%2B@192.168.1.50/Streaming/Channels/1"
Video_Path = "./data/video/resize.mp4"
#### Yolo Path ####
data_path = "./data"
weightsPath = "./weights/yolov4-custom_final.weights"
configPath = data_path + "/" + "yolov4-custom.cfg"
labelsPath = data_path + "/labels/" + "objv4.names"
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

ret, frame1 = cap.read()
ret, frame2 = cap.read()

#### ตัวแปรต่างๆ ####
cuda = True
(H,W) = (None,None)
starting_time = time.time()
frame_id = 0 
#### yolo ####
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
if cuda:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = YOLOv4(weightsPath, configPath, labelsPath, confidence_threshold=0.5, nms_threshold=0.6)

def motion_detect(frame_resize1,frame_resize2):
    diff = cv2.absdiff(frame_resize1,frame_resize2) #เปรียบเทียบ ระหว่าง frame1 และ frame2 (frame1 - frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) #แปลงภาพที่ทำการ เปรียบเทียบแล้ว เป็นภาพสีเทา
    # blur = cv2.medianBlur(gray,5)
    blur = cv2.GaussianBlur(gray, (21,21), 2) #ลบ noise
    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    kernel = np.zeros((2,2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN,kernel)
    dilated = cv2.dilate(opening, kernel, iterations=2)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    counter = 0

    for contour in cnts:
        area = cv2.contourArea(contour)
        if  area >= 3000 and area <= 7000 :
            # print(area)
            (x, y, w, h) = cv2.boundingRect(contour) 
            x2 = x + int(w/2)
            y2 = y + int(h/2)
            motion_roi = frame_resize1[y-20:y+int(h/3) , x+10:x+int(w*1.2)]
            c = x2
            if c <= (int(3*W/4+W/50)) and c >= (int(3*W/4-W/50)):
                threadProcessImage = Thread(target = model.detect(roi1))
                threadProcessImage.start()

while cap.isOpened():
    if frame1 is None and frame2 is None:
        print('Completed')
        break
    frame_id += 1
    frame_resize1 = imutils.resize(frame1, width=980) #H=551 ,W=980
    frame_resize2 = imutils.resize(frame2, width=980)
    if H is None or W is None:
        (H,W) = frame_resize1.shape[:2]
    # print(H,W)
    #### Delay
    # time.sleep(0.05)
    
    roi1 = frame_resize1[172:551, 585:865] #resize_img
    roi1_gray = cv2.cvtColor(frame_resize1, cv2.COLOR_BGR2GRAY)

    
    threadMotion= Thread(target = motion_detect(frame_resize1,frame_resize2))
    threadMotion.start()
    # diff = cv2.absdiff(frame_resize1,frame_resize2) #เปรียบเทียบ ระหว่าง frame1 และ frame2 (frame1 - frame2)
    # gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) #แปลงภาพที่ทำการ เปรียบเทียบแล้ว เป็นภาพสีเทา
    # # blur = cv2.medianBlur(gray,5)
    # blur = cv2.GaussianBlur(gray, (21,21), 2) #ลบ noise
    # _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    # kernel = np.zeros((2,2), np.uint8)
    # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel)
    # opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN,kernel)
    # dilated = cv2.dilate(opening, kernel, iterations=2)
    # cnts, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # counter = 0

    # for contour in cnts:
    #     area = cv2.contourArea(contour)
    #     if  area >= 3000 and area <= 7000 :
    #         # print(area)
    #         (x, y, w, h) = cv2.boundingRect(contour) 
    #         x2 = x + int(w/2)
    #         y2 = y + int(h/2)
    #         motion_roi = frame_resize1[y-20:y+int(h/3) , x+10:x+int(w*1.2)]
    #         c = x2
    #         if c <= (int(3*W/4+W/50)) and c >= (int(3*W/4-W/50)):
    #             threadProcessImage = Thread(target = model.detect(roi1))
    #             threadProcessImage.start()
    
    #### Line ที่ใช้แสดงพิ้นที่ในการตรวจจับ
    cv2.line(frame_resize1, (int(3*W/4+W/50),0), (int(3*W/4-W/50),1080),(255,0,0),2)
    cv2.line(frame_resize1, (int(3*W/4+W/50),0), (int(3*W/4+W/50),1080), (0, 255, 0), thickness=2)
    cv2.line(frame_resize1, (int(3*W/4-W/50),0), (int(3*W/4-W/50),1080), (0, 255, 0), thickness=2)
    

    #### FPS Show
    elapsed_time = time.time() - starting_time 
    fps = frame_id / elapsed_time
    cv2.putText(frame_resize1, "FPS: " + str(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    #### resize รูปภาพ
    # show_frame = cv2.resize(frame1,(1000,500))
    # Roi = cv2.resize(roi1,(800,500))
    #### แสดงผลการตรวจจับ
    cv2.imshow("Object Detection", frame_resize1)
    cv2.imshow("ROI", roi1)

    frame1 = frame2
    ret,frame2 = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
