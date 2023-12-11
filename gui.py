import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from main import create_animation

def submit_inputs_x():
    input1 = entry1.get()
    input2 = entry2.get()

    x_translations = np.arange(float(input1),float(input2), 0.01)
    points = np.column_stack((x_translations, np.zeros_like(x_translations)))
    create_animation(points, img, width, height, fx, fy)

def submit_inputs_y():
    input3 = entry3.get()
    input4 = entry4.get()

    y_translations = np.arange(float(input3),float(input4), 0.01)
    points = np.column_stack((np.zeros_like(y_translations), y_translations))
    create_animation(points, img, width, height, fx, fy)


# Set up Image
image_path = "data/26mmIphone13.jpg"  # Replace with the actual path to your image
img = cv2.imread(image_path)
img = img.astype(np.float32) / 255.
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, _ = img.shape

if height > 400:
    width = math.ceil(400/height * width)
    height = 400

img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)  # Adjust the size as needed

fx = 100
fy = 130
# one thing that could be cool is you have a map based on which image you select has different height,width, fx,fy vals etc.

# Create the main Tkinter window
root = tk.Tk()
root.title("Image and Input Boxes Example")

# Create a Canvas widget for displaying the image
canvas = tk.Canvas(root, width=width, height=height)
canvas.pack(pady=10)

# Load and display an image
original_image = Image.open(image_path)
resized_image = original_image.resize((width, height))  # Adjust the size as needed
photo = ImageTk.PhotoImage(resized_image)

#load image to canvas
canvas.create_image(0, 0, anchor=tk.NW, image=photo)

#put point on canvas
canvas.create_oval(width, width, width, width, fill="black", width=20)


# Create labels and Entry widgets for user input
label1 = tk.Label(root, text="x min (>-1ish):")
label1.pack(pady=5)

entry1 = tk.Entry(root, width=20)
entry1.pack(pady=5)

label2 = tk.Label(root, text="x max (<1ish):")
label2.pack(pady=5)

entry2 = tk.Entry(root, width=20)
entry2.pack(pady=5)

label3 = tk.Label(root, text="y min (>-1ish):")
label3.pack(pady=5)

entry3 = tk.Entry(root, width=20)
entry3.pack(pady=5)

label4 = tk.Label(root, text="y max (<1ish):")
label4.pack(pady=5)

entry4 = tk.Entry(root, width=20)
entry4.pack(pady=5)

# Create a button to submit the inputs
submit_button = ttk.Button(root, text="Create x animation", command=submit_inputs_x)
submit_button.pack(pady=10)

submit_button = ttk.Button(root, text="Create y animation", command=submit_inputs_y)
submit_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
