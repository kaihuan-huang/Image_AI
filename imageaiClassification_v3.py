''' Make sure you have matplotlib installed: pip install matplotlib
1. Enhance Output Format
Improve the visualization by arranging the text more neatly and adding more details like confidence scores.
2. Interactive Visualization
If you plan to run this script in an environment that supports interactivity (like Jupyter Notebooks), you can make the visualization interactive, allowing users to click through different models' predictions.
3. Detailed Analysis
Include a function to analyze and compare the performance of different models, like accuracy or time taken for predictions.
4. Improved Error Handling
Add more detailed error messages and handle specific exceptions to make debugging easier.
5. User-Friendly Interface
If this script is part of a larger application, consider creating a user-friendly interface, possibly using a framework like Tkinter (for desktop applications) or a web framework (for web applications).
6. Logging
Implement logging to keep track of the model's performance and errors, which is useful for debugging and analysis.
7. Additional Comments and Documentation
Add more comments and documentation to make the script easier to understand and modify.
8. Configurable Parameters
Allow key parameters (like model paths, image paths, number of predictions to show) to be set via command-line arguments or a configuration file.'''

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

    for model_name, model_type_function in model_types.items():
        try:
            model_path = os.path.join(execution_path, model_paths[model_name])
            predictions, probabilities, elapsed_time = load_and_classify_image(model_type_function, model_path, image_path)
            display_image_with_predictions(image_path, predictions, probabilities, model_name, elapsed_time)
        except Exception as e:
            print(f"Error using model {model_name}: {str(e)}")

if __name__ == "__main__":
    main()

# Results using ResNet50: