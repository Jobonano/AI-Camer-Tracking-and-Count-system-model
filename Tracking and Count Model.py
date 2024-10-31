# The OpenCV library for computer vision tasks.
import cv2

# A module that likely contains pre-trained models and utilities for object detection and counting.
from ultralytics import solutions

cap = cv2.VideoCapture(0) # To connect from default camera
assert cap.isOpened(), "Error reading video file"


region_points = [(100,5), (100,1000), (300,1000), (300,5)] # When connecting to Webcamj

#Adjust coordinates as needed

'''Note on Region_points:
    the first-numbers in first two bracket moves the left region line to the left/right, the higher the number the more it moves to the right
    the first-numbers in the 3&4th brackets moves the right region line to the left/rght. the higher the amount, the more it mmoves to the right 
    the second numbers in the first and the last moves it to the top. the lower the number, the hgher it is on the frame
    the second numbers in the second and the third moves it to the bottom. the higher the number, the more its bottom.'''
    


# Init Object Counter
counter = solutions.ObjectCounter(
    show=True,
    region= region_points,
    model="yolo11n.pt",
)



# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    
    
    im0 = counter.count(im0)
    #video_writer.write(im0)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), ord('Q')]:  # Quit if 'q' is pressed
        break
    
cap.release()
#video_writer.release()
cv2.destroyAllWindows()
