import cv2
import matplotlib.pyplot as plt
import numpy as np

def project3Dto2D(camera_x, camera_y, fx, fy, movex, movey):
    """
    Projects the 3D points of the inner and outer rectangle onto the 2D plane
    camera_x, camera_y are the translations of the camera, they set the translate parameters of the intrinsic matrix
    fx x fy defines the size of the inner rectangle
    movex, movey are the x and y translations of the inner rectangle
    """

    # Points of the camera rectangles as vertices in the 3D space of a 1x1x1 cube 
    # The inner rectangle is at z=1 and the outer rectangle is at z=0.3
    inner_3d = np.array([[-.5,-.5,1,1], [.5,-.5,1,1], [.5,.5,1,1], [-.5,.5,1,1]], np.float32)
    outer_3d = np.array([[-.5,-.5,0.3,1], [.5,-.5,0.3,1], [.5,.5,0.3,1], [-.5,.5,0.3,1]], np.float32)

    # Inner Rectangle, Untranslated, Bottom Left/Right, Top Right/Left
    i_u_bl, i_u_br, i_u_tr, i_u_tl = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    inner_rect_untranslated = [i_u_bl, i_u_br, i_u_tr, i_u_tl]
    # Outer Rectangle, Untranslated, Bottom Left/Right, Top Right/Left
    o_u_bl, o_u_br, o_u_tr, o_u_tl = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    outer_rect_untranslated = [o_u_bl, o_u_br, o_u_tr, o_u_tl]
    # Inner Rectangle, Translated, Bottom Left/Right, Top Right/Left
    i_t_bl, i_t_br, i_t_tr, i_t_tl = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    inner_rect_translated = [i_t_bl, i_t_br, i_t_tr, i_t_tl]
    # Outer Rectangle, Translated, Bottom Left/Right, Top Right/Left
    o_t_bl, o_t_br, o_t_tr, o_t_tl = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    outer_rect_translated = [o_t_bl, o_t_br, o_t_tr, o_t_tl]

    # Extrinsic matrix is identity when the camera is not translated
    extrinsic_matrix_untranslated = np.array([[1, 0, 0, 0], 
                                              [0, 1, 0, 0], 
                                              [0, 0, 1, 0]])

    # Use the function arguments to determine how to translate the camera
    # NOTE: The z translation can transform the camera in 3D space (going INTO the picture), but we chose not to go down the route of exploring that.
    # Instead, we focused our efforts on translating the camera in the x and y directions.
    extrinsic_matrix_translated = np.array([[1, 0, 0, camera_x], 
                                            [0, 1, 0, camera_y], 
                                            [0, 0, 1, 0]]) # type: ignore

    intrinsic_matrix= np.array([[fx, 0, movex], 
                                [0, fy, movey], 
                                [0, 0, 1]], np.float32) 
    
    # Create the camera matrices by multiplying the intrinsic and extrinsic matrices
    camera_matrix_untranslated = np.dot(intrinsic_matrix, extrinsic_matrix_untranslated)
    camera_matrix_translated = np.dot(intrinsic_matrix, extrinsic_matrix_translated)

    # Project the 3D points onto the 2D plane
    # Multiply the camera matrix by the 3D points and divide by the z coordinate to get the 2D points
    for i in range(4):
        temp = np.dot(camera_matrix_untranslated, inner_3d[i])
        inner_rect_untranslated[i] = temp / temp[2]

        temp = np.dot(camera_matrix_untranslated, outer_3d[i])
        outer_rect_untranslated[i] = temp / temp[2]

        temp = np.dot(camera_matrix_translated, inner_3d[i])
        inner_rect_translated[i] = temp / temp[2]

        temp = np.dot(camera_matrix_translated, outer_3d[i])
        outer_rect_translated[i] = temp / temp[2]

    # Fix the translation of the rectangles in the x direction because it is reversed (flips it to the other side of the scene)
    translate_fix_x = 2*(inner_rect_untranslated[0][0] - inner_rect_translated[0][0])
    for i in range(4):
        inner_rect_translated[i][0] += translate_fix_x
        outer_rect_translated[i][0] += translate_fix_x

    # Can be commented in to show if the Projection is "correct"
    # plt.plot((outer_rect_untranslated[0][0], inner_rect_untranslated[0][0]), (outer_rect_untranslated[0][1],inner_rect_untranslated[0][1]), 'r')
    # plt.plot((outer_rect_untranslated[1][0], inner_rect_untranslated[1][0]), (outer_rect_untranslated[1][1],inner_rect_untranslated[1][1]), 'r')
    # plt.plot((outer_rect_untranslated[2][0], inner_rect_untranslated[2][0]), (outer_rect_untranslated[2][1],inner_rect_untranslated[2][1]), 'r')
    # plt.plot((outer_rect_untranslated[3][0], inner_rect_untranslated[3][0]), (outer_rect_untranslated[3][1],inner_rect_untranslated[3][1]), 'r')

    # plt.plot((outer_rect_translated[0][0], inner_rect_translated[0][0]), (outer_rect_translated[0][1],inner_rect_translated[0][1]), 'b')
    # plt.plot((outer_rect_translated[1][0], inner_rect_translated[1][0]), (outer_rect_translated[1][1],inner_rect_translated[1][1]), 'b')
    # plt.plot((outer_rect_translated[2][0], inner_rect_translated[2][0]), (outer_rect_translated[2][1],inner_rect_translated[2][1]), 'b')
    # plt.plot((outer_rect_translated[3][0], inner_rect_translated[3][0]), (outer_rect_translated[3][1],inner_rect_translated[3][1]), 'b')
    # plt.imshow(img)
    # plt.show()

    return inner_rect_untranslated, outer_rect_untranslated, inner_rect_translated, outer_rect_translated


def create_side_images(img, inner_rect_pts, outer_rect_pts, w, h):
    """
    Splits the image based on the lines found between 3D points projected into the 2D plane.
    Creates masks for each "face" of the picture (background, left wall, ceiling, right wall, floor) that only include the pixels in that face.
    Returns each mask.
    """

    y,x=np.mgrid[0:h,0:w]

    # Set Outer rectangle points: top left, top right, bottom right, bottom left
    tl_out, tr_out, br_out, bl_out = outer_rect_pts[0], outer_rect_pts[1], outer_rect_pts[2], outer_rect_pts[3]

    # Set Inner rectangle points: top left, top right, bottom right, bottom left
    tl_in, tr_in, br_in, bl_in = inner_rect_pts[0], inner_rect_pts[1], inner_rect_pts[2], inner_rect_pts[3]

    plt.plot((tl_out[1],tl_in[1]), (tl_out[0],tl_in[0]), marker='o')
    # # TR LINE (1399,775) (0,353)
    plt.plot((tr_out[1],tr_in[1]), (tr_out[0],tr_in[0]), marker='o')
    # # BR LINE (1399,775) (886, 533)
    plt.plot((br_out[1],br_in[1]), (br_out[0],br_in[0]), marker='o')
    # # BL LINE (0,625) (886, 533)
    plt.plot((bl_out[1],bl_in[1]), (bl_out[0],bl_in[0]), marker='o')
    plt.imshow(img)
    plt.show()

    # Creates a mask for the inner rectangle (background)
    inner_rect_mask = (x >= tl_in[0]) & (x <= tr_in[0]) & (y >= tl_in[1]) & (y <= bl_in[1])
    inner_rect_mask = np.stack([inner_rect_mask] * img.shape[2], axis=-1).astype(np.uint8)
    inner_rect = inner_rect_mask * img

    # Creates a mask for the left panel (left wall)
    # Calculates slope and intercept to edge of image
    top_m = (tl_in[1]-tl_out[1])/(tl_in[0]-tl_out[0])
    top_intercept = tl_in[1] - (top_m * tl_in[0])
    bot_m = (bl_in[1]-bl_out[1])/(bl_in[0]-bl_out[0])
    bot_intercept = bl_in[1] - (bot_m * bl_in[0])
    left_panel_mask = (x>=0) & (x<=tl_in[0]) & (y >=top_m*x + top_intercept) & (y <=(bot_m*x) + bot_intercept)
    left_panel_mask = np.stack([left_panel_mask] * img.shape[2], axis=-1).astype(np.uint8)
    left_rect = (left_panel_mask * img)
    plt.imshow(left_rect)
    plt.show()

    # Creates a mask for the top panel (ceiling)
    left_m = (tl_in[1]-tl_out[1])/(tl_in[0]-tl_out[0])
    left_intercept = tl_in[1] - (left_m * tl_in[0])
    right_m = (tr_in[1]-tr_out[1])/(tr_in[0]-tr_out[0])
    right_intercept = tr_in[1] - (right_m * tr_in[0])
    top_panel_mask =  (y<tl_in[1]) & (y <(left_m*x) + left_intercept) & (y < (right_m*x) + right_intercept)
    top_panel_mask = np.stack([top_panel_mask] * img.shape[2], axis=-1).astype(np.uint8)
    top_rect = (top_panel_mask * img)

    # Creates a mask for the right panel (right wall)
    top_m = (tr_in[1]-tr_out[1])/(tr_in[0]-tr_out[0])
    top_intercept = tr_in[1] - (top_m * tr_in[0])
    bot_m = (br_in[1]-br_out[1])/(br_in[0]-br_out[0])
    bot_intercept = br_in[1] - (bot_m * br_in[0])
    right_panel_mask = (x>tr_in[0]) & (y > (top_m*x) + top_intercept) & (y < (bot_m*x) + bot_intercept)
    right_panel_mask = np.stack([right_panel_mask] * img.shape[2], axis=-1).astype(np.uint8)
    right_rect = (right_panel_mask * img) 

    # Creates a mask for the bottom panel (floor)
    left_m = (bl_in[1]-bl_out[1])/(bl_in[0]-bl_out[0])
    left_intercept = bl_in[1] - (left_m * bl_in[0])
    right_m = (br_in[1]-br_out[1])/(br_in[0]-br_out[0])
    right_intercept = br_in[1] - (right_m * br_in[0])
    bottom_panel_mask = (y>bl_in[1]) & (y > (left_m*x) + left_intercept) & (y > (right_m*x) + right_intercept)
    bottom_panel_mask = np.stack([bottom_panel_mask] * img.shape[2], axis=-1).astype(np.uint8)
    bottom_rect = (bottom_panel_mask * img)

    # Return each of the individual masks
    return inner_rect, left_rect, top_rect, right_rect, bottom_rect

def createHomography(old_quad, new_quad, img, width, height):
    """
    Creates a homography and warps the image given the size of the image and the corners of the quadrilaterals that correspond
    """
    M,_ = cv2.findHomography(old_quad,new_quad)
    out = cv2.warpPerspective(img,M,(int(width),int(height)))
    final_fill = np.zeros((height, width, 3))
    final_fill += out
    return out
def find_view(x_t, y_t, fx, fy, movex, movey, img, width, height):
    i_u, o_u, i_t, o_t = project3Dto2D(camera_x=x_t,camera_y=y_t, fx=fx, fy=fy, movex=movex, movey=movey)
    # Creates the masks for each of the panels 
    inner,left,top,right,bot = create_side_images(img,i_u, o_u, width, height)

    # Create homography correspondences for each of the "faces"
    old_left = np.array([o_u[0],i_u[0],i_u[3],o_u[3]])
    new_left = np.array([o_t[0],i_t[0],i_t[3],o_t[3]])

    old_top = np.array([o_u[0],i_u[0],i_u[1],o_u[1]])
    new_top = np.array([o_t[0],i_t[0],i_t[1],o_t[1]])

    old_right = np.array([o_u[1],i_u[1],i_u[2],o_u[2]])
    new_right = np.array([o_t[1],i_t[1],i_t[2],o_t[2]])

    old_bottom = np.array([o_u[3],i_u[3],i_u[2],o_u[2]])
    new_bottom = np.array([o_t[3],i_t[3],i_t[2],o_t[2]])

    old_inner = np.array([i_u[0],i_u[1],i_u[2],i_u[3]])
    new_inner = np.array([i_t[0],i_t[1],i_t[2],i_t[3]])

    # Create the homographies and warp each of the panels
    l_panel = createHomography(old_left,new_left,left, width, height)
    plt.imshow(l_panel)
    plt.show()
    t_panel = createHomography(old_top,new_top,top, width, height)
    r_panel = createHomography(old_right,new_right,right, width, height)
    b_panel = createHomography(old_bottom,new_bottom,bot, width, height)
    inner_panel = createHomography(old_inner,new_inner,inner, width, height)

    # Compose the final image by adding each of the panels together
    # If conditions prevent the panels from adding on top of each other if they go beyond the geography of the 3D cube.
    out = np.zeros_like(img)
    if x_t > -.5:
        out += r_panel
    if y_t < .5:
        out += t_panel
    if x_t < .5:
        out += l_panel
    if y_t > -.5:
        out += b_panel
    out += inner_panel

    return out
def create_animation(points, img, width, height, fx, fy, movex, movey):
    """
    Uses OpenCV to create an animation that shows the view from the camera as it moves around the 3D cube
    """
    for x_t,y_t in points:
        # If it is ever at the edge of the 3D cube, skip it to avoid division by 0
        if np.isclose(x_t,0.5) or np.isclose(x_t,-0.5) or np.isclose(y_t,0.5) or np.isclose(y_t,-0.5):
            continue 
        out = find_view(x_t, y_t, fx, fy, movex, movey, img, width, height)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imshow("window",out)
        cv2.waitKey(2)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    
    """
    Entry point for testing utility functions. Does not work with the GUI, just takes in a single image and creates an animation.
    Comment in X, Y, or Circle points to view images from those perspectives
    """

    # Change this path to "unit test" different parameters.
    img = cv2.imread("data/1.jpg")
    img = img.astype(np.float32) / 255.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, dsize=(800, 800), interpolation=cv2.INTER_CUBIC)
    height, width, _ = img.shape

    fx = 150
    fy = 120   

    # X translation animation
    # x_translations = np.arange(-0.7,0.7, 0.01)
    # points = np.column_stack((x_translations, np.zeros_like(x_translations)))
    # create_animation(points,img, width, height, fx, fy)

    #  Y Translation animation
    # y_translations = np.arange(-0.7,0.7, 0.01)
    # points = np.column_stack((np.zeros_like(y_translations), y_translations))
    # create_animation(points,img, width, height, fx, fy)
 
    # Circular Translation Animation
    theta = np.arange(0, 2*np.pi, 0.05)
    points = np.column_stack((0.3 * np.cos(theta), 0.3 * np.sin(theta)))

    # Creates inner rectangle in the center of the image. 
    create_animation(points,img, width, height, fx, fy, 1.45*width/3, 2.35*height/3)