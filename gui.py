import tkinter as tk
from PIL import Image, ImageTk

class GUI:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("TIP GUI")

        self.image = Image.open(image_path)
        self.image_tk = ImageTk.PhotoImage(self.image)

        self.canvas = tk.Canvas(root, width=self.image.width, height=self.image.height)
        self.canvas.pack()

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

        self.points = []  # Store the points of the spidery mesh

        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.root.mainloop()

    def on_canvas_click(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))
        self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="red")

if __name__ == "__main__":
    root = tk.Tk()
    image_path = "test.jpg"
    app = GUI(root, image_path)