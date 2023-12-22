from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()

prediction = ImageClassification()

# prediction.setModelTypeAsResNet50()
# prediction.setModelPath(os.path.join(execution_path, "ImageClassification_4Algorithons/resnet50-19c8e357.pth"))

# prediction.setModelTypeAsMobileNetV2()
# prediction.setModelPath(os.path.join(execution_path, "ImageClassification_4Algorithons/mobilenet_v2-b0353104.pth"))

prediction.setModelTypeAsInceptionV3()
prediction.setModelPath(os.path.join(execution_path, "ImageClassification_4Algorithons/inception_v3_google-1a9a5a14.pth"))

# prediction.setModelTypeAsDenseNet121()
# prediction.setModelPath(os.path.join(execution_path, "/Users/huanhuan/Document/03-Company/HardWare/Image_AI2/ImageClassification_4Algorithons/densenet121-a639ec97.pth"))

prediction.loadModel()
prediction.useCPU()

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "data-images/3.jpg"), result_count=10)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)·