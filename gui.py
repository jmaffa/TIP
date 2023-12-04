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

        self.numOfInnerRectGrid = 10

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
            self.canvas.create_oval(x - 2, y - 2, x + 5, y + 5, width = 0, fill="green")
            self.vanishing_point = [x, y]
            self.draw_vanishing_lines(self.numOfInnerRectGrid)
            self.draw_innergrid(self.numOfInnerRectGrid)
            # self.draw_sectiongrid(self.numOfInnerRectGrid)


            #DRAW VANISHING LINES

        print(self.points)
    def draw_vanishing_lines(self, numOfInnerRectGrid):
        topEdge_step = (self.inner_rect[2] - self.inner_rect[0])/ numOfInnerRectGrid
        sideEdge_step = (self.inner_rect[3] - self.inner_rect[1])/ numOfInnerRectGrid

        for i in range(numOfInnerRectGrid):
            self.canvas.create_line(self.inner_rect[0]+(topEdge_step*(i+1)), self.inner_rect[1], self.vanishing_point[0], self.vanishing_point[1], fill="orange")
            self.canvas.create_line(self.inner_rect[0]+(topEdge_step*(i+1)), self.inner_rect[3], self.vanishing_point[0], self.vanishing_point[1], fill="orange")
                    
            self.canvas.create_line(self.inner_rect[0], self.inner_rect[1]+(sideEdge_step*(i+1)), self.vanishing_point[0], self.vanishing_point[1],  fill="orange")
            self.canvas.create_line(self.inner_rect[2], self.inner_rect[1]+(sideEdge_step*(i+1)), self.vanishing_point[0], self.vanishing_point[1], fill="orange")

            # #SLOPES
            # slope_tl_vp = (self.vanishing_point[1] - self.inner_rect[1]) / (self.vanishing_point[0] - self.inner_rect[0]+(topEdge_step*(i+1)))

            # # slope_bl_vp = (self.vanishing_point[1] - self.inner_rect[3]) / (self.vanishing_point[0] - self.inner_rect[0]+(topEdge_step*(i+1)))

            # # slope_tr_vp = (self.vanishing_point[1] - self.inner_rect[1]) / (self.vanishing_point[0] - self.inner_rect[2])

            # # slope_br_vp = (self.vanishing_point[1] - self.inner_rect[3]) / (self.vanishing_point[0] - self.inner_rect[2])

            # # Intersection points
            # self.edge_tl = (0, self.inner_rect[1] - slope_tl_vp * self.inner_rect[0]+(topEdge_step*(i+1)))
            # # self.edge_bl = (0, self.inner_rect[3] - slope_bl_vp * self.inner_rect[0])

            # # # inner rect points to edge points
            # self.canvas.create_line(self.inner_rect[0]+(topEdge_step*(i+1)), self.inner_rect[1], self.edge_tl[0], self.edge_tl[1], fill="orange")
            # # self.canvas.create_line(self.inner_rect[0], self.inner_rect[3], self.edge_bl[0], self.edge_bl[1], fill="blue")
            

        # top left to vanishing point
        self.canvas.create_line(self.inner_rect[0], self.inner_rect[1], self.vanishing_point[0], self.vanishing_point[1],  fill="blue")
        # bot left to vanishing point
        self.canvas.create_line(self.inner_rect[0], self.inner_rect[3], self.vanishing_point[0], self.vanishing_point[1], fill="blue")
        #top right to vanishing point
        self.canvas.create_line(self.inner_rect[2], self.inner_rect[1], self.vanishing_point[0], self.vanishing_point[1], fill="blue")
        #bot right to vanishing point
        self.canvas.create_line(self.inner_rect[2], self.inner_rect[3], self.vanishing_point[0], self.vanishing_point[1], fill="blue")

        # Slopes
        slope_tl_vp = (self.vanishing_point[1] - self.inner_rect[1]) / (self.vanishing_point[0] - self.inner_rect[0])
        slope_bl_vp = (self.vanishing_point[1] - self.inner_rect[3]) / (self.vanishing_point[0] - self.inner_rect[0])
        slope_tr_vp = (self.vanishing_point[1] - self.inner_rect[1]) / (self.vanishing_point[0] - self.inner_rect[2])
        slope_br_vp = (self.vanishing_point[1] - self.inner_rect[3]) / (self.vanishing_point[0] - self.inner_rect[2])

        # Intersection points
        self.edge_tl = (0, self.inner_rect[1] - slope_tl_vp * self.inner_rect[0])
        self.edge_bl = (0, self.inner_rect[3] - slope_bl_vp * self.inner_rect[0])
        self.edge_tr = (self.max_x, self.inner_rect[1] + slope_tr_vp * (self.max_x - self.inner_rect[2]))
        self.edge_br = (self.max_x, self.inner_rect[3] + slope_br_vp * (self.max_x - self.inner_rect[2]))

        # inner rect points to edge points
        self.canvas.create_line(self.inner_rect[0], self.inner_rect[1], self.edge_tl[0], self.edge_tl[1], fill="blue")
        self.canvas.create_line(self.inner_rect[0], self.inner_rect[3], self.edge_bl[0], self.edge_bl[1], fill="blue")
        self.canvas.create_line(self.inner_rect[2], self.inner_rect[1], self.edge_tr[0], self.edge_tr[1], fill="blue")
        self.canvas.create_line(self.inner_rect[2], self.inner_rect[3], self.edge_br[0], self.edge_br[1], fill="blue")


    def draw_innergrid(self, numOfRectangleLength):
            # Draw grid lines on the inner rectangle
            for i in range(1, numOfRectangleLength):  # Adjust the number of grid lines as needed
                x_grid = self.inner_rect[0] + i * (self.inner_rect[2] - self.inner_rect[0]) / numOfRectangleLength
                self.canvas.create_line(x_grid, self.inner_rect[1], x_grid, self.inner_rect[3], fill="red")

                y_grid = self.inner_rect[1] + i * (self.inner_rect[3] - self.inner_rect[1]) / numOfRectangleLength
                self.canvas.create_line(self.inner_rect[0], y_grid, self.inner_rect[2], y_grid, fill="red")

    # def draw_sectiongrid(self, numofRectangleLengthSection):
    #     # Draw grid between lines outside of the inner rectangle
    #     s
                
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