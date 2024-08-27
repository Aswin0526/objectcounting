from ultralytics import YOLO
import cv2
from ultralytics.solutions import object_counter

model = YOLO('model.pt')

class_names = model.names

cap = cv2.VideoCapture('./apple2.mp4')

ret,frame = cap.read()

region_points = [(800,0),(800,frame.shape[0])]

counter = object_counter.ObjectCounter()

counter.set_args( view_img=True,
    reg_pts=region_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False)

    results = tracks[0].boxes.data
    im0 = counter.start_counting(im0, tracks)
    
    cv2.imshow('apple counting',im0)


cap.release()
cv2.destroyAllWindows()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model.predict(frame, save=False, conf=0.4)
#     detections = results[0].boxes.data

    

#     for detection in detections:
#         x1, y1, x2, y2, conf, class_id = detection[:6]
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         label = f"{model.names[int(class_id)]}"

#         cv2.putText(frame, f"Count-{count}", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        
       
#         cv2.line(frame, (800, 0), (800, frame.shape[0]), (255, 0, 0), thickness=2)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


