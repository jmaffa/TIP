import cv2
import matplotlib.pyplot as plt
import numpy as np

fx = 300
fy = 300

# the inner rectangle becomes the size of the fx, fy ?????
def project3Dto2D(translate_x, translate_y, fx, fy, movex, movey):
    # perhaps we need to project the image when untranslated, get the inner and outer rectangle points (R)

    # print(width)
    # print(height)

    # NOTE changing the width and height parameters moves the inner rectangle, 2.2*height/3 fits for 1_full.jpg pretty well :]
    intrinsic_matrix= np.array([[fx, 0, movex], 
                                [0, fy, movey], 
                                [0, 0, 1]], np.float32) 
    inner_3d = np.array([[-.5,-.5,1,1], [.5,-.5,1,1], [.5,.5,1,1], [-.5,.5,1,1]], np.float32)
    outer_3d = np.array([[-.5,-.5,0.3,1], [.5,-.5,0.3,1], [.5,.5,0.3,1], [-.5,.5,0.3,1]], np.float32)

    # points to define
    # untranslated 
    i_u_bl, i_u_br, i_u_tr, i_u_tl = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    o_u_bl, o_u_br, o_u_tr, o_u_tl = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    # translated
    i_t_bl, i_t_br, i_t_tr, i_t_tl = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    o_t_bl, o_t_br, o_t_tr, o_t_tl = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))

    inner_rect_untranslated = [i_u_bl, i_u_br, i_u_tr, i_u_tl]
    outer_rect_untranslated = [o_u_bl, o_u_br, o_u_tr, o_u_tl]
    inner_rect_translated = [i_t_bl, i_t_br, i_t_tr, i_t_tl]
    outer_rect_translated = [o_t_bl, o_t_br, o_t_tr, o_t_tl]

    extrinsic_matrix_untranslated = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    extrinsic_matrix_translated = np.array([[1, 0, 0, translate_x], [0, 1, 0, translate_y], [0, 0, 1, 0]])

    camera_matrix_untranslated = np.dot(intrinsic_matrix, extrinsic_matrix_untranslated)
    camera_matrix_translated = np.dot(intrinsic_matrix, extrinsic_matrix_translated)

    
    for i in range(4):
        temp = np.dot(camera_matrix_untranslated, inner_3d[i])
        inner_rect_untranslated[i] = temp / temp[2]

        temp = np.dot(camera_matrix_untranslated, outer_3d[i])
        outer_rect_untranslated[i] = temp / temp[2]

        temp = np.dot(camera_matrix_translated, inner_3d[i])
        inner_rect_translated[i] = temp / temp[2]

        temp = np.dot(camera_matrix_translated, outer_3d[i])
        outer_rect_translated[i] = temp / temp[2]

    translate_fix_x = 2*(inner_rect_untranslated[0][0] - inner_rect_translated[0][0])
    for i in range(4):
        inner_rect_translated[i][0] += translate_fix_x
        outer_rect_translated[i][0] += translate_fix_x

    # Draws lines to show if projection is "correct"
    plt.plot((outer_rect_untranslated[0][0], inner_rect_untranslated[0][0]), (outer_rect_untranslated[0][1],inner_rect_untranslated[0][1]), 'r')
    plt.plot((outer_rect_untranslated[1][0], inner_rect_untranslated[1][0]), (outer_rect_untranslated[1][1],inner_rect_untranslated[1][1]), 'r')
    plt.plot((outer_rect_untranslated[2][0], inner_rect_untranslated[2][0]), (outer_rect_untranslated[2][1],inner_rect_untranslated[2][1]), 'r')
    plt.plot((outer_rect_untranslated[3][0], inner_rect_untranslated[3][0]), (outer_rect_untranslated[3][1],inner_rect_untranslated[3][1]), 'r')

    plt.plot((outer_rect_translated[0][0], inner_rect_translated[0][0]), (outer_rect_translated[0][1],inner_rect_translated[0][1]), 'b')
    plt.plot((outer_rect_translated[1][0], inner_rect_translated[1][0]), (outer_rect_translated[1][1],inner_rect_translated[1][1]), 'b')
    plt.plot((outer_rect_translated[2][0], inner_rect_translated[2][0]), (outer_rect_translated[2][1],inner_rect_translated[2][1]), 'b')
    plt.plot((outer_rect_translated[3][0], inner_rect_translated[3][0]), (outer_rect_translated[3][1],inner_rect_translated[3][1]), 'b')

    return inner_rect_untranslated, outer_rect_untranslated, inner_rect_translated, outer_rect_translated


def create_side_images(img, inner_rect_pts, outer_rect_pts, w, h):
      
    y,x=np.mgrid[0:h,0:w]

    tl_out = outer_rect_pts[0]
    # print(tl_out)
    tr_out = outer_rect_pts[1]
    # print(tr_out)
    br_out = outer_rect_pts[2]
    # print(br_out)
    bl_out = outer_rect_pts[3]
    # print(bl_out)

    tl_in = inner_rect_pts[0]
    tr_in = inner_rect_pts[1]
    br_in = inner_rect_pts[2]
    bl_in = inner_rect_pts[3]

    inner_rect_mask = (x >= tl_in[0]) & (x <= tr_in[0]) & (y >= tl_in[1]) & (y <= bl_in[1])
    inner_rect_mask = np.stack([inner_rect_mask] * img.shape[2], axis=-1).astype(np.uint8)
    inner_rect = inner_rect_mask * img

    # TODO: ADD THEM but MAYBE DO THAT AT END BUT ACTUALLY JUST DO THAT TO CHECK

    # left panel : CORRECT
    top_m = (tl_in[1]-tl_out[1])/(tl_in[0]-tl_out[0])
    top_intercept = tl_in[1] - (top_m * tl_in[0])
    bot_m = (bl_in[1]-bl_out[1])/(bl_in[0]-bl_out[0])
    bot_intercept = bl_in[1] - (bot_m * bl_in[0])
    left_panel_mask = (x>=0) & (x<=tl_in[0]) & (y >=top_m*x + top_intercept) & (y <=(bot_m*x) + bot_intercept)
    left_panel_mask = np.stack([left_panel_mask] * img.shape[2], axis=-1).astype(np.uint8)
    left_rect = (left_panel_mask * img)
    # plt.imshow(left_rect)

    # top panel : ??
    left_m = (tl_in[1]-tl_out[1])/(tl_in[0]-tl_out[0])
    left_intercept = tl_in[1] - (left_m * tl_in[0])
    
    right_m = (tr_in[1]-tr_out[1])/(tr_in[0]-tr_out[0])
    right_intercept = tr_in[1] - (right_m * tr_in[0])

    top_panel_mask =  (y<tl_in[1]) & (y <(left_m*x) + left_intercept) & (y < (right_m*x) + right_intercept)
    top_panel_mask = np.stack([top_panel_mask] * img.shape[2], axis=-1).astype(np.uint8)
    top_rect = (top_panel_mask * img)
    # plt.imshow(top_rect)

    # right panel : CORRECT
    top_m = (tr_in[1]-tr_out[1])/(tr_in[0]-tr_out[0])
    top_intercept = tr_in[1] - (top_m * tr_in[0])
    bot_m = (br_in[1]-br_out[1])/(br_in[0]-br_out[0])
    bot_intercept = br_in[1] - (bot_m * br_in[0])
    right_panel_mask = (x>tr_in[0]) & (y > (top_m*x) + top_intercept) & (y < (bot_m*x) + bot_intercept)
    right_panel_mask = np.stack([right_panel_mask] * img.shape[2], axis=-1).astype(np.uint8)
    right_rect = (right_panel_mask * img) 
    # plt.imshow(right_rect)

    # bottom panel: TODO
    left_m = (bl_in[1]-bl_out[1])/(bl_in[0]-bl_out[0])
    left_intercept = bl_in[1] - (left_m * bl_in[0])
    right_m = (br_in[1]-br_out[1])/(br_in[0]-br_out[0])
    right_intercept = br_in[1] - (right_m * br_in[0])
    bottom_panel_mask = (y>bl_in[1]) & (y > (left_m*x) + left_intercept) & (y > (right_m*x) + right_intercept)
    bottom_panel_mask = np.stack([bottom_panel_mask] * img.shape[2], axis=-1).astype(np.uint8)
    bottom_rect = (bottom_panel_mask * img)
    # plt.imshow(bottom_rect)
    

    new = np.zeros_like(img)
    new+= inner_rect+left_rect + top_rect + right_rect + bottom_rect

    return inner_rect, left_rect, top_rect, right_rect, bottom_rect

def createHomography(old_quad, new_quad, img, width, height):
    M,_ = cv2.findHomography(old_quad,new_quad)
    out = cv2.warpPerspective(img,M,(int(width),int(height)))
    final_fill = np.zeros((height, width, 3))
    final_fill += out
    return out
def create_animation(points, img, width, height, fx, fy, movex, movey):
    for x_t,y_t in points:
        if np.isclose(x_t,0.5) or np.isclose(x_t,-0.5) or np.isclose(y_t,0.5) or np.isclose(y_t,-0.5):
            continue 
        i_u, o_u, i_t, o_t = project3Dto2D(translate_x=x_t,translate_y=y_t, fx=fx, fy=fy, movex=movex, movey=movey)
        inner,left,top,right,bot = create_side_images(img,i_u, o_u, width, height)
        old_left = np.array([o_u[0],i_u[0],i_u[3],o_u[3]])
        new_left = np.array([o_t[0],i_t[0],i_t[3],o_t[3]])

        old_top= np.array([o_u[0],i_u[0],i_u[1],o_u[1]])
        new_top = np.array([o_t[0],i_t[0],i_t[1],o_t[1]])

        old_right = np.array([o_u[1],i_u[1],i_u[2],o_u[2]])
        new_right = np.array([o_t[1],i_t[1],i_t[2],o_t[2]])

        old_bottom = np.array([o_u[3],i_u[3],i_u[2],o_u[2]])
        new_bottom = np.array([o_t[3],i_t[3],i_t[2],o_t[2]])

        old_inner = np.array([i_u[0],i_u[1],i_u[2],i_u[3]])
        new_inner = np.array([i_t[0],i_t[1],i_t[2],i_t[3]])

        l_panel = createHomography(old_left,new_left,left, width, height)
        t_panel = createHomography(old_top,new_top,top, width, height)
        r_panel = createHomography(old_right,new_right,right, width, height)
        b_panel = createHomography(old_bottom,new_bottom,bot, width, height)
        inner_panel = createHomography(old_inner,new_inner,inner, width, height)

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
        # out+= l_panel+t_panel+r_panel+b_panel+inner_panel

        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imshow("window",out)
        cv2.waitKey(2)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img = cv2.imread("data/26mmIphone13.jpg")
    img = img.astype(np.float32) / 255.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(800, 800), interpolation=cv2.INTER_CUBIC)

    height, width, _ = img.shape

    # X translation animation
    x_translations = np.arange(-0.7,0.7, 0.01)
    points = np.column_stack((x_translations, np.zeros_like(x_translations)))
    # create_animation(points,img, width, height, fx, fy)

    #  Y Translation animation
    y_translations = np.arange(-0.7,0.7, 0.01)
    points = np.column_stack((np.zeros_like(y_translations), y_translations))
    # create_animation(points,img, width, height, fx, fy)
 
    # Circular Translation Animation
    theta = np.arange(0, 2*np.pi, 0.05)
    points = np.column_stack((0.3 * np.cos(theta), 0.3 * np.sin(theta)))
    
    create_animation(points,img, width, height, fx, fy, width/2, height/2)