import matplotlib.pyplot as plt
import numpy as np
import gui

# Define the vertices of a cube
vertices = np.array([
    [-1, -1, -1],  # 0
    [1, -1, -1],   # 1
    [1, 1, -1],    # 2
    [-1, 1, -1],   # 3
    [-1, -1, 1],   # 4
    [1, -1, 1],    # 5
    [1, 1, 1],     # 6
    [-1, 1, 1]     # 7
])

# Define the indices of the vertices that form the cube's faces
faces = np.array([
    [0, 1, 2, 3],  # Bottom face
    [4, 5, 6, 7],  # Top face
    [0, 4, 7, 3],  # Left face
    [1, 5, 6, 2],  # Right face
    [0, 1, 5, 4],  # Front face
    [2, 3, 7, 6]   # Back face
])

# Create a 2D image with inner and outer rectangles
fig, ax = plt.subplots()

# Plot the cube faces
for face in faces:
    ax.plot(vertices[face, 0], vertices[face, 1], color='gray', linestyle='-', linewidth=2)

# Draw rectangles on the image (you can adjust the coordinates based on your requirements)
outer_rectangle = plt.Rectangle((-1.5, -1.5), 3, 3, fill=None, edgecolor='red', linewidth=2)
inner_rectangle = plt.Rectangle((-0.5, -0.5), 1, 1, fill=None, edgecolor='blue', linewidth=2)

ax.add_patch(outer_rectangle)
ax.add_patch(inner_rectangle)

# Set axis limits
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])

# Set aspect ratio to equal
ax.set_aspect('equal', 'box')

# Display the plot
plt.show()