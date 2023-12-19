'''1. Interactive User Input
Allow users to select the image file through a file dialog instead of hardcoding the image path. This can be done using tkinter.filedialog.
2. Model Selection Interface
Instead of running all models sequentially, provide an option for users to select a specific model to use for classification.
3. Improved Result Display
Enhance the display of results by improving the layout, such as using subplots to show the original image alongside a bar chart of probabilities.
4. Batch Processing
Add functionality to process multiple images in a folder, displaying and comparing results for each image.
5. Performance Metrics
Display additional performance metrics, like accuracy or inference time, if you have the ground truth available for the images.
6. Error Handling and Logging
Implement more robust error handling and logging for better debugging and user feedback.
7. Configurable Parameters
Allow key parameters (like number of predictions to return) to be configurable either through command-line arguments or a configuration file.
8. Saving Results
Provide an option to save the classification results and images with annotated predictions to a file.'''

import tkinter as tk
from tkinter import filedialog
from imageai.Classification import ImageClassification
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

def load_and_classify_image(model_type_function, model_path, image_path):
    prediction = ImageClassification()
    model_type_function(prediction)
    prediction.setModelPath(model_path)
    prediction.loadModel()
    start_time = time.time()
    predictions, probabilities = prediction.classifyImage(image_path, result_count=5)
    end_time = time.time()
    return predictions, probabilities, end_time - start_time

def display_image_with_predictions(image_path, predictions, probabilities, model_name, elapsed_time):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Model: {model_name} (Time Taken: {elapsed_time:.2f} seconds)")

    for i, (prediction, probability) in enumerate(zip(predictions, probabilities)):
        plt.text(0, 50 + i*20, f"{prediction} : {probability:.2f}%", color='blue')

    plt.show()


def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp"), ("All files", "*.*")])
    root.destroy()
    return file_path

def main():
    execution_path = os.getcwd()
    image_path = os.path.join(execution_path, "data-images/3.jpg")

    model_types = {
        "ResNet50": ImageClassification.setModelTypeAsResNet50,
        "MobileNetV2": ImageClassification.setModelTypeAsMobileNetV2,
        "InceptionV3": ImageClassification.setModelTypeAsInceptionV3,
        "DenseNet121": ImageClassification.setModelTypeAsDenseNet121
    }

    model_paths = {
        "ResNet50": "ImageClassification_4Algorithons/resnet50-19c8e357.pth",
        "MobileNetV2": "ImageClassification_4Algorithons/mobilenet_v2-b0353104.pth",
        "InceptionV3": "ImageClassification_4Algorithons/inception_v3_google-1a9a5a14.pth",
        "DenseNet121": "ImageClassification_4Algorithons/densenet121-a639ec97.pth"
    }

    image_path = select_image()  # Let the user select an image
    if not image_path:  # Check if an image was selected
        print("No image selected.")
        return

    for model_name, model_type_function in model_types.items():
        try:
            model_path = os.path.join(execution_path, model_paths[model_name])
            predictions, probabilities, elapsed_time = load_and_classify_image(model_type_function, model_path, image_path)
            display_image_with_predictions(image_path, predictions, probabilities, model_name, elapsed_time)
        except Exception as e:
            print(f"Error using model {model_name}: {str(e)}")

if __name__ == "__main__":
    main()
