import cv2
import numpy as np
import imutils
import time
from threading import Thread

from numpy.core.fromnumeric import sort
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
    print("เปิดไฟล์วีดีโอ..")
    time.sleep(0.5)
    print("สำเร็จ")

ret, frame1 = cap.read()
ret, frame2 = cap.read()

#### ตัวแปรต่างๆ ####
cuda = True
(H,W) = (None,None)
(roi_h,roi_w) = (None,None)
starting_time = time.time()
frame_id = 0
counter = 0
first_frame = None
#### yolo ####
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
if cuda:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
model = YOLOv4(weightsPath, configPath, labelsPath, confidence_threshold=0.5, nms_threshold=0.6)

while cap.isOpened():
    if frame1 is None and frame2 is None:
        print('Completed')
        break
    frame_id += 1
    frame_resize1 = imutils.resize(frame1, width=800)
    frame_resize2 = imutils.resize(frame2, width=800)
    if H is None or W is None:
        (H,W) = frame1.shape[:2]
    #### Delay
    # time.sleep(0.05)
    
    # roi1 = frame1[527:1047, 1316: 1665]
    roi1 = frame_resize1[526:1047,954:1750]
    # if roi_h is None or roi_w is None:
    #     (roi_h,roi_w) = roi1.shape[:2]
    # ปรับแต่ง
    diff = cv2.absdiff(frame_resize1,frame_resize2) #เปรียบเทียบ ระหว่าง frame1 และ frame2 (frame1 - frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) #แปลงภาพที่ทำการ เปรียบเทียบแล้ว เป็นภาพสีเทา
    # blur = cv2.medianBlur(gray,5)
    blur = cv2.GaussianBlur(gray, (21,21), 2) #ลบ noise
    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    kernel = np.zeros((2,2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN,kernel)
    # opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN,kernel,iterations=3)
    dilated = cv2.dilate(opening, kernel, iterations=2)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    

    for contour in cnts:
        area = cv2.contourArea(contour)
        if  area >= 2500 and area <= 7000 :
            (x, y, w, h) = cv2.boundingRect(contour)
            x2 = x + int(w/2)
            y2 = y + int(h/2)
            cv2.rectangle(frame_resize1,(x,y),(x+w,y+h),(0,255,0),3)
            # cv2.circle(frame1, (x2,y2), 4, (0,255,0), -1)
            c = x2
            # print(w/h)
            if c <= (int(3*W/4+W/50)) and c >= (int(3*W/4-W/50)):
                print('เข้า')
                counter += 1
                threadProcessImage = Thread(target = model.detect(roi1))
                threadProcessImage.start()
                
    cv2.line(frame_resize1, (int(3*W/6+W/50),0), (int(3*W/6-W/50),1080),(255,0,0),2)
    cv2.line(frame_resize1, (int(3*W/6+W/50),0), (int(3*W/6+W/50),1080), (0, 255, 0), thickness=2)
    cv2.line(frame_resize1, (int(3*W/6-W/50),0), (int(3*W/6-W/50),1080), (0, 255, 0), thickness=2)
    


    elapsed_time = time.time() - starting_time 
    fps = frame_id / elapsed_time
    cv2.putText(frame1, "FPS: " + str(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    #### resize
    # show_frame = cv2.resize(frame1,(1000,500))
    # Roi = cv2.resize(roi1,(800,500))
    #### show
    cv2.imshow("Object Detection", frame_resize1)
    cv2.imshow("ROI", dilated)

    frame1 = frame2
    ret,frame2 = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
