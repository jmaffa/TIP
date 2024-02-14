import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import math
from util import create_animation

def submit_inputs_x():
    """
    Creates an animation of translating the camera along the x axis 
    """
    # Processes Inputs
    input1 = entry1.get()
    input2 = entry2.get()
    if input1 == "" or input2 == "":
        print("Please enter valid inputs")
        return
    if input2 < input1:
        print("Min must be less than Max")
        return
    
    # Creates tuples of x,y translations to animate
    x_translations = np.arange(float(input1),float(input2), 0.01)
    points = np.column_stack((x_translations, np.zeros_like(x_translations)))

    if(innerRectCreated and vanishingPtcreated):
        rectx1, recty1, rectx2,recty2 = canvas.coords(inner_rect) # type: ignore
        fx = rectx2 - rectx1 
        fy = recty2 - recty1
        create_animation(points, img, width, height, fx, fy, movex, movey)
    else:
        print("Choose your back plane!")

def submit_inputs_y():
    """
    Creates an animation of translating the camera along the y axis
    """
    # Processes Inputs
    input3 = entry3.get()
    input4 = entry4.get()
    if input3 == "" or input4 == "":
        print("Please enter valid inputs")
        return
    if input4 < input3:
        print("Min must be less than Max")
        return
    
    # Creates tuples of x,y translations to animate
    y_translations = np.arange(float(input3),float(input4), 0.01)
    points = np.column_stack((np.zeros_like(y_translations), y_translations))

    if(innerRectCreated and vanishingPtcreated):
        rectx1, recty1, rectx2,recty2 = canvas.coords(inner_rect) # type: ignore
        fx = rectx2 - rectx1 
        fy = recty2 - recty1
        create_animation(points, img, width, height, fx, fy, movex, movey)
    else:
        print("Choose your back plane!")

def circular_animation():
    """
    Creates an animation of translating the camera along a circular path
    """
    # Creates tuples of x,y translations to animate
    theta = np.arange(0, 2*np.pi, 0.05)
    points = np.column_stack((0.2 * np.cos(theta), 0.2 * np.sin(theta)))

    if(innerRectCreated and vanishingPtcreated):
        rectx1, recty1, rectx2,recty2 = canvas.coords(inner_rect) # type: ignore
        fx = rectx2 - rectx1 
        fy = recty2 - recty1
        create_animation(points, img, width, height, fx, fy, movex, movey)
    else:
        print("Choose your back plane and vanishing point!")

def clear_shapes():
    """
    Removes any shapes on the canvas that were used to define the inner rectangle
    """
    global innerRectCreated,vanishingPtcreated

    for shape in canvas.find_withtag("shape"):
        canvas.delete(shape)
    innerRectCreated = False
    vanishingPtcreated = False

def prepare_img(file_path):
    """
    Sets up the image for the canvas (returned) and for the transformation (stored as global)
    """
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
    return resized_image
def open_image():
    """
    Opens an image from the file system and displays it on the canvas
    """
    global photo, innerRectCreated, vanishingPtcreated  # Make img, width, and height global
    file_path = filedialog.askopenfilename()
    if file_path:
        resized_image  = prepare_img(file_path)
        # Load and display the new image
        photo = ImageTk.PhotoImage(resized_image)
        # Clear previous image
        canvas.delete("all")
        # Load new image to canvas
        canvas.create_image(0, 0, anchor=tk.NW, image=photo) # type: ignore
        innerRectCreated = False
        vanishingPtcreated = False

def on_click(event):
    """
    Handles logic to define the inner rectangle in the image. Uses tkinter canvas and calculates rectangle from top left and bottom right corners
    """
    global start_x, start_y,inner_rect, innerRectCreated,vanishingPtcreated,movex,movey

    if(innerRectCreated is False):
        if start_x is None and start_y is None:
            # First click, store the starting coordinates
            start_x, start_y = event.x, event.y
            point_size = 3
            canvas.create_oval(start_x - point_size, start_y - point_size, start_x + point_size, start_y + point_size, fill="red", tags="shape")
        else:
            # Second click, draw the rectangle based on bottom right coordinate
            end_x, end_y = event.x, event.y
            point_size = 3
            canvas.create_oval(end_x - point_size, end_y - point_size, end_x + point_size, end_y + point_size, fill="red", tags="shape")

            inner_rect = canvas.create_rectangle(start_x, start_y, end_x, end_y, outline="black",tags="shape") # type: ignore
            innerRectCreated = True

            # Reset starting coordinates for the next rectangle
            start_x, start_y = None, None
    else:
        # Third click, draw the vanishing point
        if(vanishingPtcreated == False):
            movex, movey = event.x, event.y
            point_size = 3
            canvas.create_oval(movex - point_size, movey - point_size, movex + point_size, movey + point_size, fill="red", tags=["shape", "vanishingpt"])
            vanishingPtcreated = True

if __name__ == "__main__":
    start_x, start_y = None, None
    innerRectCreated = False
    vanishingPtcreated = False
    inner_rect = None

    # Set up Image
    image_path = "data/1.jpg" 
    img = cv2.imread(image_path)
    img = img.astype(np.float32) / 255.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    if height > 400:
        width = math.ceil(400/height * width)
        height = 400 
    img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)  

    root = tk.Tk()
    root.title("Tour Into Picture (TIP)")

    # Create a Canvas widget for displaying the image
    canvas = tk.Canvas(root, width=width, height=height)
    canvas.pack(pady=10, padx=10)

    # Load and display an image as a clickable Canvas
    original_image = Image.open(image_path)
    resized_image = original_image.resize((width, height)) 
    photo = ImageTk.PhotoImage(resized_image)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.bind('<Button-1>', on_click) # type: ignore

    frame = tk.Frame(root)
    frame.pack(pady=20)

    # Create labels and Entry widgets for user input for x and y translations
    label1 = tk.Label(frame, text="x min (>-1ish):")
    label1.grid(row=1, column=0, padx=2, pady=5)
    entry1 = tk.Entry(frame, width=5)
    entry1.grid(row=1, column=1, padx=2, pady=5)
    label2 = tk.Label(frame, text="x max (<1ish):")
    label2.grid(row=1, column=2, padx=2, pady=5)
    entry2 = tk.Entry(frame, width=5)
    entry2.grid(row=1, column=3, padx=2, pady=5)
    submit_x_button = ttk.Button(frame, text="Create x animation", command=submit_inputs_x)
    submit_x_button.grid(row=1, column=4, padx=2, pady=5)

    label3 = tk.Label(frame, text="y min (>-1ish):")
    label3.grid(row=2, column=0, padx=2, pady=5)
    entry3 = tk.Entry(frame, width=5)
    entry3.grid(row=2, column=1, padx=2, pady=5)
    label4 = tk.Label(frame, text="y max (<1ish):")
    label4.grid(row=2, column=2, padx=2, pady=5)
    entry4 = tk.Entry(frame, width=5)
    entry4.grid(row=2, column=3, padx=2, pady=5)
    submit_y_button = ttk.Button(frame, text="Create y animation", command=submit_inputs_y)
    submit_y_button.grid(row=2, column=4, padx=2, pady=5)

    # Button for circular animation
    circle_button = ttk.Button(frame, text="Circle Animation", command=circular_animation)
    circle_button.grid(row=3, column=2, padx=2, pady=5)

    # Button to clear the inner rectangle
    clear_button = ttk.Button(frame, text="Reset Back Plane", command=clear_shapes)
    clear_button.grid(row=4, column=2, padx=2, pady=5)

    # Button to open another image
    open_button = ttk.Button(frame, text="Open Image", command=open_image)
    open_button.grid(row=5, column=2, padx=2, pady=5)

    root.mainloop()