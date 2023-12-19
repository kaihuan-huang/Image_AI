import tkinter as tk
from tkinter import filedialog

def select_image_and_save_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    root.destroy()

    if file_path:
        with open("selected_image_path.txt", "w") as file:
            file.write(file_path)

if __name__ == "__main__":
    select_image_and_save_path()
