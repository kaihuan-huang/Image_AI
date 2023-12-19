'''To enhance your script for object detection using ImageAI, you can integrate several improvements and additional features. These enhancements can make your script more flexible, user-friendly, and powerful. Here's how you can upgrade it:

### 1. Interactive Image Selection
   - Allow users to select the image for detection through a file dialog interface, making the script more interactive.

### 2. Display Detection Results Graphically
   - Use Matplotlib or another library to display the detected image with bounding boxes and labels directly in the script.

### 3. Batch Processing
   - Add functionality to process multiple images in a folder, displaying results for each image.

### 4. Model Selection
   - Provide an option for users to select between different detection models (RetinaNet, YOLOv3, TinyYOLOv3).

### 5. Custom Object Detection
   - Add functionality for detecting only specific objects from the available classes.

### 6. Save Detection Results
   - Give users the option to save the detected images and detection data (like bounding box coordinates and class names) to files.

### 7. GUI Integration
   - If you want to make the script more accessible, consider integrating it into a GUI using Tkinter or another GUI framework.

### 8. Performance Metrics
   - Display additional metrics, such as the processing time for each image.

### 9. Advanced Visualization
   - Enhance visualization by showing detection confidence scores and implementing more sophisticated graphics.'''

#  Needed to be fixed: Program crashes (crashes



import tkinter as tk
from tkinter import filedialog
from imageai.Detection import ObjectDetection
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    root.destroy()
    return file_path

def display_detected_objects(image_path, detections):
    # Load image
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw bounding boxes
    for eachObject in detections:
        box_points = eachObject["box_points"]
        rect = patches.Rectangle((box_points[0], box_points[1]), box_points[2] - box_points[0], box_points[3] - box_points[1], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(box_points[0], box_points[1], f"{eachObject['name']} : {eachObject['percentage_probability']:.2f}%", color='white', fontsize=8, backgroundcolor='red')

    plt.show()

def main():
    execution_path = os.getcwd()

    # Initialize ObjectDetection
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path, "ObjectDetectionModel/yolov3.pt"))
    detector.loadModel()

    # Select and detect objects in the image
    image_path = select_image()
    if image_path:
        detections = detector.detectObjectsFromImage(input_image=image_path, output_image_path=os.path.join(execution_path, "imagenew.jpg"), minimum_percentage_probability=30)
        display_detected_objects(image_path, detections)

if __name__ == "__main__":
    main()


'''- This script uses a file dialog to select an image and Matplotlib to display the image with detected objects.
- The `display_detected_objects` function draws bounding boxes and labels on the detected objects.
- Ensure all necessary libraries are installed (`ImageAI`, `Matplotlib`, `PIL`).
- Run this script in an environment that supports GUI operations, as it opens file dialogs and image windows.'''