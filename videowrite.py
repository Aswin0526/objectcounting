import cv2 

cap = cv2.VideoCapture('./apple.mp4')

ret,frame = cap.read()

if not ret:
    exit()

output_file = cv2.VideoWriter('outputapple.avi',cv2.VideoWriter_fourcc(*'XVID'),20,(frame.shape[1],frame.shape[0]))

while(cap.isOpened()):

    ret,frame = cap.read()

    if not ret:
        break

    output_file.write(frame)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
output_file.release()
cv2.destroyAllWindows()


