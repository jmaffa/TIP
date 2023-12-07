import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # full image
    # background image
    # foreground image
    img = cv2.imread("data/1_full.jpg")
    background = cv2.imread("data/1_background.jpg")
    foreground = cv2.imread("data/1_foreground.jpg")

    img = img.astype(np.float32) / 255.
    background = background.astype(np.float32) / 255.
    foreground = foreground.astype(np.float32) / 255.

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    
    plt.imshow(img)
    plt.show()
    plt.imshow(background)
    plt.show()
    plt.imshow(foreground)
    plt.show()


# need an ima
def create_rects(img, x,y,w,h):
    plt.draw
    pass
def set_vanishing_point():
    pass
def set_foreground_billboards():
    pass

