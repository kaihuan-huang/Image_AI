'''Create a GUI for Image Classification:

Install Tkinter:

Tkinter is usually included with Python. If it's not installed, you can install it using your package manager (like apt on Ubuntu: sudo apt-get install python3-tk).
Basic GUI Structure:

Import Tkinter and other necessary libraries.
Create a main window.
Add widgets like buttons, labels, and image display areas.
Functionality:

Implement functions to browse and select images.
Add options to select the model for classification.
Display the classification results and the selected image.
Running the Script:

Run the script normally, and the GUI should appear, allowing users to interact with your image classification application.'''
import tkinter as tk
from tkinter import ttk  # For Combobox
from tkinter import filedialog  # For file selection
from PIL import Image, ImageTk  # For image display
import os

# Your existing functions
from imageai.Classification import ImageClassification

def load_and_classify_image(model_type_function, model_path, image_path):
    prediction = ImageClassification()
    model_type_function(prediction)
    prediction.setModelPath(model_path)
    prediction.loadModel()
    start_time = time.time()
    predictions, probabilities = prediction.classifyImage(image_path, result_count=5)
    end_time = time.time()
    return predictions, probabilities, end_time - start_time

def select_image():
    global panel_image  # Global reference for the image

    file_types = [("Image files", "*.jpg *.jpeg *.png *.gif *.bmp"), ("All files", "*.*")]
    file_path = filedialog.askopenfilename(filetypes=file_types)
    if file_path:
        try:
            img = Image.open(file_path)
            img.thumbnail((500, 500), Image.ANTIALIAS)
            panel_image = ImageTk.PhotoImage(img)  # Use the global variable
            panel.configure(image=panel_image)
            panel.image = panel_image  # Keep a reference

            # Perform classification and update result label
            predictions, probabilities = classify_image(file_path)
            result_text = "\n".join(f"{pred} : {prob:.2f}%" for pred, prob in zip(predictions, probabilities))
            result_label.config(text=result_text)
        except Exception as e:
            print(f"Error loading image: {e}")
            result_label.config(text="Error loading image.")


def main():
    global panel, model_selector, result_label
    window = tk.Tk()
    window.title("Image Classification")

    # Add Model selection
    model_label = tk.Label(window, text="Select Model:")
    model_label.pack(padx=10, pady=5)

    model_names = ["ResNet50", "MobileNetV2", "InceptionV3", "DenseNet121"]
    model_selector = ttk.Combobox(window, values=model_names)
    model_selector.pack(padx=10, pady=5)
    # Result label
    result_label = tk.Label(window, text="Classification Results", justify=tk.LEFT)
    result_label.pack(padx=10, pady=10)


    # Run the application
    window.mainloop()

if __name__ == "__main__":
    main()
