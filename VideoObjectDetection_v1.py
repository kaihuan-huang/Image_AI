'''ImageAI provides convenient, flexible and powerful methods to perform object detection on videos. The video object detection class provided only supports RetinaNet, YOLOv3 and TinyYOLOv3. This version of ImageAI provides commercial grade video objects detection features, which include but not limited to device/IP camera inputs, per frame, per second, per minute and entire video analysis for storing in databases and/or real-time visualizations and for future insights.

To start performing video object detection, you must download the RetinaNet, YOLOv3 or TinyYOLOv3 object detection model via the links below:

RetinaNet (Size = 130 mb, high performance and accuracy, with longer detection time)
YOLOv3 (Size = 237 mb, moderate performance and accuracy, with a moderate detection time)
TinyYOLOv3 (Size = 34 mb, optimized for speed and moderate performance, with fast detection time)
Because video object detection is a compute intensive tasks, we advise you perform this experiment using a computer with a NVIDIA GPU and the GPU version of Tensorflow installed. Performing Video Object Detection CPU will be slower than using an NVIDIA GPU powered computer. You can use Google Colab for this experiment as it has an NVIDIA K80 GPU available for free.'''



from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "/Users/huanhuan/Documents/03-Company/HardWare/ImageAI_v3/ObjectDetectionModel/retinanet_resnet50_fpn_coco-eeacb38b.pth"))
detector.loadModel()



video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "data-videos/traffic.mp4"),
                                output_file_path=os.path.join(execution_path, "traffic_detected")
                                , frames_per_second=20, log_progress=True)
print(video_path)