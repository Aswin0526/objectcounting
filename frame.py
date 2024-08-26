import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('./apple.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    
    cv2.line(frame,(1240,0),(1240,715),(255,0,0),thickness=2)
    plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    plt.show()
    

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
