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
        self.outer_rect = [(0, 0), (self.max_x, 0), (self.max_x, self.max_y), (0, self.max_y)]
        self.cube_vertices = []
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
            self.create_cube()
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
            # self.canvas.create_line(self.inner_rect[0]+(topEdge_step*(i+1)), self.inner_rect[3], self.vanishing_point[0], self.vanishing_point[1], fill="orange")
                    
            self.canvas.create_line(self.inner_rect[0], self.inner_rect[1]+(sideEdge_step*(i+1)), self.vanishing_point[0], self.vanishing_point[1],  fill="orange")
            # self.canvas.create_line(self.inner_rect[2], self.inner_rect[1]+(sideEdge_step*(i+1)), self.vanishing_point[0], self.vanishing_point[1], fill="orange")

            # #SLOPES
            slope_tl_vp_denominator = (self.vanishing_point[0] - (self.inner_rect[0]+(topEdge_step*(i+1))))
            if slope_tl_vp_denominator == 0:
                print("slope is inf")
                self.edge_tl = (0, self.inner_rect[0]+(topEdge_step*(i+1)))
            else:
                print("no inf slope")
                slope_tl_vp = (self.vanishing_point[1] - self.inner_rect[1]) / slope_tl_vp_denominator
                print(slope_tl_vp)
                # slope_bl_vp = (self.vanishing_point[1] - self.inner_rect[3]) / (self.vanishing_point[0] - (self.inner_rect[0]+(topEdge_step*(i+1))))

                if slope_tl_vp < 0:
                    print("LESS THAN 0")
                    self.edge_tl = (0, self.inner_rect[1] - (-slope_tl_vp * self.inner_rect[0]+(topEdge_step*(i+1))))
                else:
                    self.edge_tl = (0, (self.inner_rect[1] - (slope_tl_vp * self.inner_rect[0]+(topEdge_step*(i+1)))))


            # # Intersection points
            self.edge_tl = (0, self.inner_rect[1] - (slope_tl_vp * self.inner_rect[0]+(topEdge_step*(i+1))))
            # self.edge_bl = (0, self.inner_rect[3] - abs(slope_bl_vp) * self.inner_rect[0])

            # # inner rect points to edge points
            self.canvas.create_line(self.inner_rect[0]+(topEdge_step*(i+1)), self.inner_rect[1], self.edge_tl[0], self.edge_tl[1], fill="orange")
            # self.canvas.create_line(self.inner_rect[0]+(topEdge_step*(i+1)), self.inner_rect[3], self.edge_bl[0], self.edge_bl[1], fill="orange")

    

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
        # slope_bl_vp = (self.vanishing_point[1] - self.inner_rect[3]) / (self.vanishing_point[0] - self.inner_rect[0])
        # slope_tr_vp = (self.vanishing_point[1] - self.inner_rect[1]) / (self.vanishing_point[0] - self.inner_rect[2])
        # slope_br_vp = (self.vanishing_point[1] - self.inner_rect[3]) / (self.vanishing_point[0] - self.inner_rect[2])

        # Intersection points
        self.edge_tl = (0, self.inner_rect[1] - slope_tl_vp * self.inner_rect[0])
        # self.edge_bl = (0, self.inner_rect[3] - slope_bl_vp * self.inner_rect[0])
        # self.edge_tr = (self.max_x, self.inner_rect[1] + slope_tr_vp * (self.max_x - self.inner_rect[2]))
        # self.edge_br = (self.max_x, self.inner_rect[3] + slope_br_vp * (self.max_x - self.inner_rect[2]))

        # inner rect points to edge points
        self.canvas.create_line(self.inner_rect[0], self.inner_rect[1], self.edge_tl[0], self.edge_tl[1], fill="blue")
        # self.canvas.create_line(self.inner_rect[0], self.inner_rect[3], self.edge_bl[0], self.edge_bl[1], fill="blue")
        # self.canvas.create_line(self.inner_rect[2], self.inner_rect[1], self.edge_tr[0], self.edge_tr[1], fill="blue")
        # self.canvas.create_line(self.inner_rect[2], self.inner_rect[3], self.edge_br[0], self.edge_br[1], fill="blue")

    def intersection_with_y_axis(self, slope, b):
        # Solve for x
        x_intersection = -b / slope
        
        return x_intersection

    def draw_innergrid(self, numOfRectangleLength):
            # Draw grid lines on the inner rectangle
            for i in range(1, numOfRectangleLength):  # Adjust the number of grid lines as needed
                x_grid = self.inner_rect[0] + i * (self.inner_rect[2] - self.inner_rect[0]) / numOfRectangleLength
                self.canvas.create_line(x_grid, self.inner_rect[1], x_grid, self.inner_rect[3], fill="red")

                y_grid = self.inner_rect[1] + i * (self.inner_rect[3] - self.inner_rect[1]) / numOfRectangleLength
                self.canvas.create_line(self.inner_rect[0], y_grid, self.inner_rect[2], y_grid, fill="red")

    def calculate_cube_vertices(self):
        # Extract points from the inner and outer rectangles
        # [(0, 0), (self.max_x, 0), (self.max_x, self.max_y), (0, self.max_y)]
        # [x1, y1, x2, y2]
        outer_top_left, outer_top_right, outer_bottom_right, outer_bottom_left = self.outer_rect[0], self.outer_rect[1], self.outer_rect[2], self.outer_rect[3]
        inner_rect_width = self.inner_rect[2] - self.inner_rect[0]
        inner_rect_height = self.inner_rect[3] - self.inner_rect[1]
        inner_top_left, inner_top_right, inner_bottom_left, inner_bottom_right = (self.inner_rect[0], self.inner_rect[1]), \
                                                                    ((self.inner_rect[0] + inner_rect_width), self.inner_rect[1]),  \
                                                                    (self.inner_rect[0], (self.inner_rect[1] + inner_rect_height)),  \
                                                                    ((self.inner_rect[0] + inner_rect_width), (self.inner_rect[1] + inner_rect_height))
            
        # Calculate cube vertices
        self.cube_vertices = [
            outer_top_left,
            outer_bottom_left,
            outer_top_right,
            outer_bottom_right,
            inner_top_left,
            inner_bottom_left,
            inner_top_right,
            inner_bottom_right
        ]

    def create_cube(self):

        # Calculate cube vertices based on inner and outer rectangles
        self.calculate_cube_vertices()

        # Draw the cube
        self.canvas.create_line(self.cube_vertices[0], self.cube_vertices[1], fill="green")
        self.canvas.create_line(self.cube_vertices[0], self.cube_vertices[2], fill="green")
        self.canvas.create_line(self.cube_vertices[2], self.cube_vertices[3], fill="green")
        self.canvas.create_line(self.cube_vertices[1], self.cube_vertices[3], fill="green")

        # self.canvas.create_line(self.cube_vertices[4], self.cube_vertices[5], fill="green")
        # self.canvas.create_line(self.cube_vertices[5], self.cube_vertices[6], fill="green")
        # self.canvas.create_line(self.cube_vertices[6], self.cube_vertices[7], fill="green")
        # self.canvas.create_line(self.cube_vertices[7], self.cube_vertices[4], fill="green")

        self.canvas.create_line(self.cube_vertices[0], self.cube_vertices[4], fill="green")
        self.canvas.create_line(self.cube_vertices[1], self.cube_vertices[5], fill="green")
        self.canvas.create_line(self.cube_vertices[2], self.cube_vertices[6], fill="green")
        self.canvas.create_line(self.cube_vertices[3], self.cube_vertices[7], fill="green")

        # List to store the 3D coordinates of the points
        points_3d = []

if __name__ == "__main__":
    root = tk.Tk()
    # image_path = "data/1_full.jpg"
    label = tk.Label(text="Name")
    entry = tk.Entry()
    # app = GUI(root, image_path)
    