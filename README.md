# Object Detection and Counting with OpenCV and YOLO

## Overview

This Jupyter Notebook demonstrates how to utilize the OpenCV library in conjunction with the Ultraytics `solutions` module to perform real-time object detection and counting via a webcam feed. It employs a pre-trained YOLO (You Only Look Once) model to identify and count objects within a specified region of interest (ROI).

## Requirements

To run this notebook, you need to have the following libraries installed:

- OpenCV
- Ultralytics (for the YOLO model)

You can install these packages using pip:

```bash
pip install opencv-python ultralytics
```

## Code Explanation

### 1. Import Libraries

```python
import cv2
from ultralytics import solutions
```

- `cv2`: This is the OpenCV library used for computer vision tasks.
- `solutions`: This module from Ultraytics contains pre-trained models and utilities for object detection.

### 2. Initialize Webcam

```python
cap = cv2.VideoCapture(0) # To connect from default camera
assert cap.isOpened(), "Error reading video file"
```

This section establishes a connection to the default webcam. An assertion checks if the webcam is successfully opened.

### 3. Define Region of Interest (ROI)

```python
region_points = [(100,5), (100,1000), (300,1000), (300,5)]
```

The `region_points` variable defines the coordinates of the ROI where objects will be counted. The comments provide guidance on how to adjust these coordinates based on your needs.

### 4. Initialize Object Counter

```python
counter = solutions.ObjectCounter(
    show=True,
    region=region_points,
    model="yolo11n.pt",
)
```

Here, the `ObjectCounter` is initialized with the specified ROI and YOLO model. The `show` parameter indicates that the output will be displayed in real-time.

### 5. Process Video Frames

```python
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    im0 = counter.count(im0)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), ord('Q')]:  # Quit if 'q' is pressed
        break
```

In this loop, frames are captured from the webcam, processed for object detection and counting, and displayed. The loop continues until the user presses 'q' or the webcam feed is no longer available.

### 6. Cleanup

```python
cap.release()
cv2.destroyAllWindows()
```

After exiting the loop, the code releases the webcam and closes all OpenCV windows.

## Usage

1. Ensure that your webcam is connected.
2. Run the notebook cell containing the above code.
3. Adjust the `region_points` as necessary to define the area for object counting.
4. Press 'q' to stop the video feed and exit the application.

## Notes

- Make sure the YOLO model file (`yolo11n.pt`) is in the correct directory or specify the full path.
- You may need to adjust the coordinates in `region_points` based on your specific use case.

## Conclusion

This notebook provides a foundational approach to real-time object detection and counting using OpenCV and YOLO. You can further enhance this application by implementing features such as saving the results, integrating more complex models, or processing video files instead of live feed.
