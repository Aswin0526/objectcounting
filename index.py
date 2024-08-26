from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture('./apple.mp4')

count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, save=False, conf=0.4)
    detections = results[0].boxes.data

    

    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection[:6]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{model.names[int(class_id)]}"

        cv2.putText(frame, f"Count-{count}", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        
       
        cv2.line(frame, (800, 0), (800, frame.shape[0]), (255, 0, 0), thickness=2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
