import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from util import create_animation

def submit_inputs_x():
    input1 = entry1.get()
    input2 = entry2.get()
    x_translations = np.arange(float(input1),float(input2), 0.01)
    points = np.column_stack((x_translations, np.zeros_like(x_translations)))

    if(innerRectCreated):
        rectx1, recty1, rectx2,recty2 = canvas.coords(inner_rect)
        fx = rectx2 - rectx1 
        fy = recty2 - recty1
        create_animation(points, img, width, height, fx, fy, movex, movey)
    else:
        print("Choose your back plane!")

def submit_inputs_y():
    input3 = entry3.get()
    input4 = entry4.get()
    y_translations = np.arange(float(input3),float(input4), 0.01)
    points = np.column_stack((np.zeros_like(y_translations), y_translations))

    if(innerRectCreated):
        rectx1, recty1, rectx2,recty2 = canvas.coords(inner_rect)
        fx = rectx2 - rectx1 
        fy = recty2 - recty1
        create_animation(points, img, width, height, fx, fy, movex, movey)
    else:
        print("Choose your back plane!")

def circular_animation():
    # Circular Translation Animation
    theta = np.arange(0, 2*np.pi, 0.05)
    points = np.column_stack((0.2 * np.cos(theta), 0.2 * np.sin(theta)))

    if(innerRectCreated):
        rectx1, recty1, rectx2,recty2 = canvas.coords(inner_rect)
        fx = rectx2 - rectx1 
        fy = recty2 - recty1
        create_animation(points, img, width, height, fx, fy, movex, movey)
    else:
        print("Choose your back plane!")

def on_click(event):
    global start_x, start_y,inner_rect, innerRectCreated,vanishingPtcreated,movex,movey

    if(innerRectCreated is False):
        if start_x is None and start_y is None:
            # First click, store the starting coordinates
            start_x, start_y = event.x, event.y
            point_size = 3
            canvas.create_oval(start_x - point_size, start_y - point_size, start_x + point_size, start_y + point_size, fill="red", tags="shape")
        else:
            # Second click, draw the rectangle
            end_x, end_y = event.x, event.y
            point_size = 3
            canvas.create_oval(end_x - point_size, end_y - point_size, end_x + point_size, end_y + point_size, fill="red", tags="shape")
            #delete points for rect
            # clear_shapes()

            inner_rect = canvas.create_rectangle(start_x, start_y, end_x, end_y, outline="black",tags="shape")
            innerRectCreated = True

            # Reset starting coordinates for the next rectangle
            start_x, start_y = None, None
    else:

        if(vanishingPtcreated == False):
            movex, movey = event.x, event.y
            point_size = 3
            canvas.create_oval(movex - point_size, movey - point_size, movex + point_size, movey + point_size, fill="red", tags=["shape", "vanishingpt"])
            vanishingPtcreated = True



def clear_shapes():
    global innerRectCreated,vanishingPtcreated
    # Clear only the shapes on the canvas (excluding the image)
    for shape in canvas.find_withtag("shape"):
     canvas.delete(shape)

    innerRectCreated = False
    vanishingPtcreated = False
def prepare_img(file_path):
    global img, width, height
    img = cv2.imread(file_path)
    img = img.astype(np.float32) / 255.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    if height > 400:
        width = math.ceil(400 / height * width)
        height = 400
    img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    original_image = Image.open(file_path)
    resized_image = original_image.resize((width, height))
    return resized_image, width, height
def open_image():
    global photo  # Make img, width, and height global
    file_path = filedialog.askopenfilename()
    if file_path:
        resized_image, width, height  = prepare_img(file_path)
        # Load and display the new image
          # Adjust the size as needed
        photo = ImageTk.PhotoImage(resized_image)
        # Clear previous image
        canvas.delete("all")

        # Load new image to canvas
        canvas.create_image(0, 0, anchor=tk.NW, image=photo) # type: ignore
        # canvas.pack(expand=True)
if __name__ == "__main__":
    start_x, start_y = None, None
    innerRectCreated = False
    vanishingPtcreated = False
    inner_rect = None

    # Set up Image
    image_path = "data/1_full.jpg"  # Replace with the actual path to your image
    img = cv2.imread(image_path)
    img = img.astype(np.float32) / 255.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    if height > 400:
        width = math.ceil(400/height * width)
        height = 400 
    img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)  # Adjust the size as needed

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
    # canvas.pack(expand=True)

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


    canvas.bind('<Button-1>', on_click)


    # #This is currently hard to determine
    movex = width/2
    movey = height/2

    # Create a button to submit the inputs
    submit_button = ttk.Button(root, text="Create x animation", command=submit_inputs_x)
    submit_button.pack(side=tk.LEFT, padx=(20, 20), pady=10, anchor=tk.CENTER, expand=True)

    submit_button = ttk.Button(root, text="Create y animation", command=submit_inputs_y)
    submit_button.pack(side=tk.LEFT, padx=(15, 20), pady=10, anchor=tk.CENTER, expand=True)

    submit_button = ttk.Button(root, text="Circle Animation", command=circular_animation)
    submit_button.pack(side=tk.LEFT, padx=(10, 20), pady=10, anchor=tk.CENTER, expand=True)


    # Create a button to clear back plane selection
    clear_button = ttk.Button(root, text="Reset Back Plane", command=clear_shapes)
    clear_button.pack(side=tk.LEFT, padx=(5, 20), pady=10, anchor=tk.CENTER, expand=True)

    open_button = ttk.Button(root, text="Open Image", command=open_image)
    open_button.pack(side=tk.LEFT, padx=(5, 20), pady=10, anchor=tk.CENTER, expand=True)


    # Start the Tkinter event loop
    root.mainloop()