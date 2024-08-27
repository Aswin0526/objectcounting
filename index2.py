from ultralytics import YOLO
from tracker import Tracker
import cv2

model = YOLO('model.pt')
class_list = model.names

tracker = Tracker()
count = 0

cap = cv2.VideoCapture('./apple2.mp4')

while True:
    ret,frame = cap.read()
    if not ret:
        break

    count+=1
    frame = cv2.resize(frame,(1020,500))

    results = model.predict(frame,save=False,conf=0.4)

    data = results[0].boxes.data

    lst = []

    for d in data:
        x1,y1,x2,y2,conf,class_id = d[:6]
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y1)
        class_id = int(class_id)
        conf = int(conf)

        if 'apple'==class_list[class_id]:
            lst.append([x1,y1,x2,y2])

    bbox_id = tracker.update(lst)
    print(bbox_id)
    for bbox in bbox_id:
        x3,y3,x4,y4,id = bbox
        cx = int(x3+x4)//2
        cy = int(y3+y4)//2
        cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
        cv2.putText(frame,str(id),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    
 
    cv2.imshow('frame',frame)

cv2.release()
cv2.destroyAllWindows()



