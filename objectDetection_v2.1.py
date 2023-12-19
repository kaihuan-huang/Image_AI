'''为了解决您的Python脚本中的闪退问题，我们可以考虑一些潜在的解决方案。重要的是要逐步排除可能导致问题的因素，以下是一些修改建议：

### 1. 分离图形界面和图像处理
图形界面（Tkinter）和图像处理（ImageAI和Matplotlib）可能存在冲突。尝试将它们分开执行，以确定问题的根源。例如，您可以首先使用Tkinter选择图像，然后关闭Tkinter窗口，再进行图像处理。

### 2. 延迟图像处理
在使用Tkinter选择图像后，添加一些延迟或将图像处理代码放在一个单独的函数中，以避免Tkinter和图像处理库之间的潜在冲突。

### 3. 检查图像路径
确保从Tkinter返回的图像路径是有效的，并且图像文件可以被ImageAI正确处理。
- 这个修改将Tkinter用于选择图像和ImageAI用于对象检测的部分分开，以减少它们之间的直接交互。
- 添加了`time.sleep(1)`来确保Tkinter界面完全关闭后再开始图像处理。这有助于避免由于GUI线程和图像处理线程之间的交互导致的问题。
- 如果仍然遇到闪退问题，请考虑将问题分开处理。首先只运行Tkinter代码，然后单独运行图像检测部分，看看问题是否仍然存在。'''




import tkinter as tk
from tkinter import filedialog
from imageai.Detection import ObjectDetection
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import time

def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    root.destroy()
    return file_path

def display_detected_objects(image_path, detections):
    # 延迟处理，确保Tkinter窗口已关闭
    time.sleep(1)
    
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

def process_image(image_path):
    execution_path = os.getcwd()

    # Initialize ObjectDetection
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path, "ObjectDetectionModel/yolov3.pt"))
    detector.loadModel()

    # # Detect objects in the image
    # if image_path:
    #     detections = detector.detectObjectsFromImage(input_image=image_path, output_image_path=os.path.join(execution_path, "imagenew.jpg"), minimum_percentage_probability=30)
    #     display_detected_objects(image_path, detections)

def main():
    image_path = select_image()
    process_image(image_path)

if __name__ == "__main__":
    main()