import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def submit_inputs():
    input1 = entry1.get()
    input2 = entry2.get()
    input3 = entry3.get()

    # Do something with the inputs, for example, print them
    print("Input 1:", input1)
    print("Input 2:", input2)
    print("Input 3:", input3)

# Create the main Tkinter window
root = tk.Tk()
root.title("Image and Input Boxes Example")

# Load and display an image
image_path = "data/1_full.jpg"  # Replace with the actual path to your image
original_image = Image.open(image_path)
resized_image = original_image.resize((600, 400))  # Adjust the size as needed
photo = ImageTk.PhotoImage(resized_image)
label_image = tk.Label(root, image=photo)
# label_image.image = photo  # Keep a reference to the image to prevent garbage collection
label_image.pack(pady=10)

# Create labels and Entry widgets for user input
label1 = tk.Label(root, text="Enter something for Box 1:")
label1.pack(pady=5)

entry1 = tk.Entry(root, width=20)
entry1.pack(pady=5)

label2 = tk.Label(root, text="Enter something for Box 2:")
label2.pack(pady=5)

entry2 = tk.Entry(root, width=20)
entry2.pack(pady=5)

label3 = tk.Label(root, text="Enter something for Box 3:")
label3.pack(pady=5)

entry3 = tk.Entry(root, width=20)
entry3.pack(pady=5)

# Create a button to submit the inputs
submit_button = ttk.Button(root, text="Submit", command=submit_inputs)
submit_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
