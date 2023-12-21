'''To upgrade the `video_path` so that the output of the object detection is stored in a new folder called "Video Object DetectionOutPut", you need to modify the `output_file_path` parameter in the `detectObjectsFromVideo` function. This parameter specifies the location and the base name of the output file.

Here's how you can modify your code:

1. **Create the Output Folder**:
   First, make sure that the "Video Object DetectionOutPut" folder exists. If it doesn't, you need to create it. You can do this either manually or by adding code to create it if it doesn't exist.

2. **Set the Output Path**:
   Modify the `output_file_path` to include the new folder and the desired output file base name.
, `output_directory` is set to the "Video Object DetectionOutPut" folder within the current execution path. The `output_file_path` in the `detectObjectsFromVideo` function is updated to use this new directory. 

Also, make sure that the specified paths in `execution_path` and `output_directory` are correct and accessible in your environment.'''

from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

# Ensure the output directory exists
output_directory = os.path.join(execution_path, "Video Object DetectionOutPut")
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

detector = VideoObjectDetection()

# ModelTypeAsRetinaNet
'''
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "/Users/huanhuan/Documents/03-Company/HardWare/ImageAI_v3/ObjectDetectionModel/retinanet_resnet50_fpn_coco-eeacb38b.pth"))
'''

# ModelTypeAsYOLOv3
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "/Users/huanhuan/Documents/03-Company/HardWare/ImageAI_v3/ObjectDetectionModel/retinanet_resnet50_fpn_coco-eeacb38b.pth"))

detector.setModelPath(os.path.join(execution_path, "/Users/huanhuan/Documents/03-Company/HardWare/ImageAI_v3/VideoObjectDetectionModels/yolov3 (1).pt"))
detector.loadModel()

input_video = os.path.join(execution_path, "data-videos/Real Estate Agent Showing to a Young Couple.mp4")
output_video = os.path.join(output_directory, "Agent Showing_detected_mini")

video_path = detector.detectObjectsFromVideo(input_file_path=input_video,
                                              output_file_path=output_video,
                                              frames_per_second=20, log_progress=True)
print(video_path)
