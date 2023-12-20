'''

1. Displaying the Output Video:
   - Since Python scripts typically run in a terminal or a command-line interface which cannot play video, you would need an external tool or a Python package that can open and play video files. One common approach is to use `OpenCV` to display the video.

2. Analyzing the Output:
   - To analyze the output, you might want to gather statistics or data from the detection process, such as the number of objects detected, the types of objects, or how often certain objects appear.

First, make sure you have `opencv-python` installed. You can install it using pip:

```bash
pip install opencv-python

Press 'q' to close the window. 
'''


from imageai.Detection import VideoObjectDetection
import os
import cv2  # Import OpenCV

execution_path = os.getcwd()

# Output directory
output_directory = os.path.join(execution_path, "Video Object DetectionOutPut")
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Set up detector
detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
model_path = os.path.join(execution_path, "/Users/huanhuan/Documents/03-Company/HardWare/ImageAI_v3/VideoObjectDetectionModels/retinanet_resnet50_fpn_coco-eeacb38b (1).pth")
detector.setModelPath(model_path)
detector.loadModel()

input_video = os.path.join(execution_path, "Image_AI/data-videos/holo1.mp4")
output_video = os.path.join(output_directory, "traffic-mini_detected")

# Detect objects in video
video_path = detector.detectObjectsFromVideo(input_file_path=input_video,
                                             output_file_path=output_video,
                                             frames_per_second=20, log_progress=True)

# Display the output video (basic example)
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Video Object Detection holo v1.2', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
