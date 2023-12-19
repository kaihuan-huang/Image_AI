from imageai.Detection import ObjectDetection
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def display_detected_objects(image_path, detections):
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for eachObject in detections:
        box_points = eachObject["box_points"]
        rect = patches.Rectangle((box_points[0], box_points[1]), box_points[2] - box_points[0], box_points[3] - box_points[1], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(box_points[0], box_points[1], f"{eachObject['name']} : {eachObject['percentage_probability']:.2f}%", color='white', fontsize=8, backgroundcolor='red')
    
    plt.show()

def perform_object_detection():
    execution_path = os.getcwd()
    image_path = ""

    with open("selected_image_path.txt", "r") as file:
        image_path = file.read().strip()

    if image_path:
        detector = ObjectDetection()
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(os.path.join(execution_path, "ObjectDetectionModel/yolov3.pt"))
        detector.loadModel()
        
        detections = detector.detectObjectsFromImage(input_image=image_path, output_image_path=os.path.join(execution_path, "imagenew.jpg"), minimum_percentage_probability=30)
        display_detected_objects(image_path, detections)

if __name__ == "__main__":
    perform_object_detection()
