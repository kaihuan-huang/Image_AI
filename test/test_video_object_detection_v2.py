Certainly! I'll provide an updated version of your script, ensuring the model paths are correctly set based on the provided model file locations:

```python
import os, sys
from typing import List
from numpy import ndarray
from os.path import dirname
from mock import patch
sys.path.insert(1, os.path.join(dirname(dirname(os.path.abspath(__file__)))))

from imageai.Detection import VideoObjectDetection

# Set the base folder for models and videos
base_folder = "/Users/huanhuan/Documents/03-Company/HardWare/ImageAI_v3"

test_folder = dirname(os.path.abspath(__file__))

video_file = os.path.join(test_folder, "data-videos", "traffic-micro.mp4")
video_file_output = os.path.join(test_folder, "data-videos", "traffic-micro-detected")


class CallbackFunctions:
    # ... (your callback methods here)


def delete_cache(files: List[str]):
    # ... (your delete_cache function here)


def test_video_detection_retinanet():
    delete_cache([video_file_output + ".mp4"])

    detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet()
    model_path = os.path.join(base_folder, "VideoObjectDetectionModels", "retinanet_resnet50_fpn_coco-eeacb38b (1).pth")
    detector.setModelPath(model_path=model_path)
    detector.loadModel()
    # ... (rest of your function here)


def test_video_detection_yolov3():
    delete_cache([video_file_output + ".mp4"])

    detector = VideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    model_path = os.path.join(base_folder, "VideoObjectDetectionModels", "yolov3 (1).pt")
    detector.setModelPath(model_path=model_path)
    detector.loadModel()
    # ... (rest of your function here)


# Define other test functions similarly, adjusting the model paths as needed


# Example of running a specific test
if __name__ == "__main__":
    test_video_detection_retinanet()
    # Call other test functions as needed
```

In this updated script, I have set the `base_folder` to the path where your model files are located (`/Users/huanhuan/Documents/03-Company/HardWare/ImageAI_v3`). The model paths in the test functions are then constructed using this base path and the specific model file names you provided. 

Please ensure that the model file names and their locations are correct as per your system's setup. If you have additional models or different file names, you should adjust the paths in the respective functions accordingly.