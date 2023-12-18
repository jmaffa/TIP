# Tour Into the Picture
## Alice Min, Joe Maffa, Serena Pulopot
This is an implementation of [Tour Into the Picture](http://graphics.cs.cmu.edu/courses/15-463/2006_fall/www/Papers/TIP.pdf). The goal of this project is to abstract an image into three dimensional space then project that onto a two dimensional plane in order to visualize an image as if it were three dimensional. In doing so, you are able to explore different "angles" of the image by placing your camera at different locations in relation to the scene. The paper also discusses a method for separating foreground objects from the background of a scene in order to reduce the warping that occurs on the foreground subjects when different angles are explored, but we chose to omit that step for time.

## Overview of our code
`main.py` contains the main logic for our GUI. It allows people to manually specify an inner rectangle around which the transformation is made. It also generates animations based on this inner rectangle and user specified parameters.

`util.py` contains the logic for the Tour into the Picture.

## How to run the project
Run `python main.py` to open up the GUI. Start by defining an inner rectangle by clicking the image first where the top left  and then bottom right corners should be. This will interpolate a rectangle from those points. Then click again to define the vanishing point within this rectangle. This defines where the rectangle will be within the image. This inner rectangle acts as the "back face" of the rectangular visualization that is created. If you would like to respecify this for any reason, choose "Reset Back Plane."

Once the inner rectangle is specified, the user can make translations in the x or y direction by entering minimum and maximum translations into the input fields then clicking the respective "Create" buttons. You can also generate an animation that follows a circular path.

In order to load an image, choose "Open Image" and select an image.There are multiple possible images to use in the `/data` directory, but feel free to use your own as well.

## Possible Setup Required
This project was completed within the cs1290_env. It also requires Tkinter so begin by installing that either with `pip install tk` or `brew reinstall python-tk@<PYTHON_VERSION>`. For example, if you are running Python 3.10, this might look like `brew reinstall python-tk@3.10`



