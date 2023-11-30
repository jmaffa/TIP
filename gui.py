import tkinter as tk
from PIL import Image, ImageTk

class GUI:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("TIP GUI")

        self.image = Image.open(image_path)
        self.image_tk = ImageTk.PhotoImage(self.image)
        self.min_x = 0
        self.min_y = 0
        self.max_x = self.image.width
        self.max_y = self.image.height

        self.canvas = tk.Canvas(root, width=self.max_x, height=self.max_y)
        self.canvas.pack()

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

        self.points = []  # Store the points of the spidery mesh
        
        
        self.inner_rect = [] # [x1, y1, x2, y2]
        self.vanishing_point = [] # [x, y]
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.root.mainloop()

    def on_canvas_click(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))
        self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="red")
        if len(self.points) == 2:
            # creates a rectangle from the top left and bottom right corners
            print("CREATING RECTANGLE")
            self.canvas.create_rectangle(self.points[0][0], self.points[0][1], self.points[1][0], self.points[1][1], outline="red")
            self.inner_rect = [self.points[0][0], self.points[0][1], self.points[1][0], self.points[1][1]]
        if len(self.points) == 3:
            print("VANISHING POINT")
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="green")
            self.vanishing_point = [x, y]
            self.draw_vanishing_lines()

            #DRAW VANISHING LINES

        print(self.points)
    def draw_vanishing_lines(self):
        # top left to vanishing point
        self.canvas.create_line(self.inner_rect[0], self.inner_rect[1], self.vanishing_point[0], self.vanishing_point[1],  fill="blue")
        # bot left to vanishing point
        self.canvas.create_line(self.inner_rect[0], self.inner_rect[3], self.vanishing_point[0], self.vanishing_point[1], fill="blue")
        #top right to vanishing point
        self.canvas.create_line(self.inner_rect[2], self.inner_rect[1], self.vanishing_point[0], self.vanishing_point[1], fill="blue")
        #bot right to vanishing point
        self.canvas.create_line(self.inner_rect[2], self.inner_rect[3], self.vanishing_point[0], self.vanishing_point[1], fill="blue")

        # TO FIND THE EDGE POINTS, FIND THE SLOPE OF EACH OF THE LINES THEN FIND WHERE IT INTERSECTS THE MAX VALUES AND STUFF
        


# VANISHING POINT MODE
# INNER RECTANGLE MODE
# OUTER RECTANGLE MODE
    # outer rectangle is determined as line from the corners of the inner rectangle to the vanishing point
    #outer rectangle is always the corners of the photo


#things to define
# vanishing point
# inner rectangle corners
if __name__ == "__main__":
    root = tk.Tk()
    image_path = "data/1_full.jpg"
    app = GUI(root, image_path)