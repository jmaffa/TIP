import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt

# Load image and normalize
C = image.imread('./data/test.jpg') / 255

xp, yp, zchannel = C.shape

new_shape = (int(xp/2), int(yp/2), zchannel)  # Preserve the number of color channels
resized_img = np.resize(C, new_shape)

xp, yp, zchannel = resized_img.shape


# Create meshgrid
x = np.arange(0, xp, 1)
y = np.arange(0, yp, 1)
Y, X = np.meshgrid(y, x)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(projection='3d')
ax.dist = 6.2
ax.view_init(elev=38, azim=-45)


# Plot surfaces using meshgrid coordinates
print("Image shape:", C.shape)
print("X shape:", X.shape)
print("Y shape:", Y.shape)


# Plot surfaces using meshgrid coordinates
ax.plot_surface(X, Y, np.full_like(X, yp), facecolors=C,
                rstride=1, cstride=1,
                antialiased=True, shade=False)

ax.plot_surface(np.full_like(X, xp), X, Y, facecolors=np.fliplr(C.transpose((1, 0, 2))),
                rstride=1, cstride=1,
                antialiased=True, shade=False)

ax.plot_surface(X, np.full_like(X, yp), Y, facecolors=np.fliplr(C.transpose((0, 1, 2))),
                rstride=1, cstride=1, antialiased=True, shade=False)

plt.show()
