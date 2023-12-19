# ImageAI/imageaiClassification_v2.py Conpare 4 model types for image classification
from imageai.Classification import ImageClassification
import os

def load_and_classify_image(model_type_function, model_path, image_path):
    prediction = ImageClassification()
    model_type_function(prediction)  # Set the model type
    prediction.setModelPath(model_path)
    prediction.loadModel()

    predictions, probabilities = prediction.classifyImage(image_path, result_count=5)
    return predictions, probabilities

def main():
    execution_path = os.getcwd()
    image_path = os.path.join(execution_path, "data-images/4.jpg")

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
            predictions, probabilities = load_and_classify_image(model_type_function, model_path, image_path)
            print(f"Results using {model_name}:")
            for eachPrediction, eachProbability in zip(predictions, probabilities):
                print(eachPrediction, ":", eachProbability)
            print("\n")
        except Exception as e:
            print(f"Error using model {model_name}: {str(e)}")

if __name__ == "__main__":
    main()
