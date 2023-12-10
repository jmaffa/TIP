import cv2
import matplotlib.pyplot as plt
import numpy as np



fx = 150
fy = 180
width = 0
height = 0


# the inner rectangle becomes the size of the fx, fy ?????
def project3Dto2D(img):
    new_img = np.zeros((height,width,3))
    # [fx 0 w/2]
    # [0 fy h/2]
    # [0 0 1]
    intrinsic_matrix= np.array([[fx, 0, width/2], 
                                [0, fy, height/2], 
                                [0, 0, 1]], np.float32) 
    
    # ???
    dist_coeffs = np.zeros((5, 1), np.float32) 

    # THOUGHT: top left inner rect. PROJECTS TO: top right inner rect
    # CV2 returns [[[1000.   143.5]]]
    # WE RETURN: 1000, 143.5
    top_right_inner = np.array([[.5],[-.5],[1]])
    top_right_outer = np.array([[.5],[-.5],[0]])

    # x,y,z = (.5,-.5,1) 

    # THOUGHT: bottom right PROJECTS TO: bottom left
    bot_left_inner = np.array([[-.5],[.5],[1]])
    bot_left_outer = np.array([[-.5],[.5],[0]])
    # x,y,z = (-.5,.5,1)

    # THOUGHT: bottom left PROJECTS TO: top left
    top_left_inner = np.array([[-.5],[-.5],[1]])
    top_left_outer = np.array([[-.5],[.5],[0]])
    # x,y,z = (-.5,-.5,1)

    # THOUGHT: top right PROJECTS TO: bottom right
    bot_right_inner = np.array([[.5],[.5],[1]])
    bot_right_outer = np.array([[-.5],[.5],[0]])
    # x,y,z = (.5,.5,1)

    # DEDUCE THAT THIS SHOULD BE TOP RIGHT OUTER RECT = imgwidth, 0
    # x,y,z = .5,-.5,-1

    # point_3d = np.vstack(([x],[y],[z]))
    # point_3d = np.array([[x,y,z]], np.float32)

    # extrinsic but also they are needed for the cv2 function
    rotation = np.zeros((3, 1), np.float32) 
    # rotation = np.array([[0.2],[0],[0]])
    # translation = np.zeros((3, 1), np.float32) 
    translation = np.array([[1.0],[1.0],[0]])


    # does matrix mult, then divides by z and returns an x,y?
    top_right_2d_inner, _ = cv2.projectPoints(top_right_inner, rotation, translation, intrinsic_matrix, dist_coeffs) 
    # top_right_2d_outer, _ = cv2.projectPoints(top_right_outer, rotation, translation, intrinsic_matrix, dist_coeffs) 
    top_left_2d_inner, _ = cv2.projectPoints(top_left_inner, rotation, translation, intrinsic_matrix, dist_coeffs) 
    # top_left_2d_outer, _ = cv2.projectPoints(top_left_outer, rotation, translation, intrinsic_matrix, dist_coeffs) 
    # print(top_left_2d_outer)
    bot_right_2d_inner, _ = cv2.projectPoints(bot_right_inner, rotation, translation, intrinsic_matrix, dist_coeffs) 
    # bot_right_2d_outer, _ = cv2.projectPoints(bot_right_outer, rotation, translation, intrinsic_matrix, dist_coeffs) 
    bot_left_2d_inner, _ = cv2.projectPoints(bot_left_inner, rotation, translation, intrinsic_matrix, dist_coeffs) 
    # bot_left_2d_outer, _ = cv2.projectPoints(bot_left_outer, rotation, translation, intrinsic_matrix, dist_coeffs) 


    # we may need ot project the 3d outer rect points

    top_right_2d_x = int(top_right_2d_inner[0][0][0])
    top_right_2d_y = int(top_right_2d_inner[0][0][1])
    # new_img[top_right_2d_y,top_right_2d_x, :] = img[570,750,:]

    top_left_2d_x = int(top_left_2d_inner[0][0][0])
    top_left_2d_y = int(top_left_2d_inner[0][0][1])
    # new_img[top_left_2d_y,top_left_2d_x, :] = img[570,650,:]

    bot_right_2d_x = int(bot_right_2d_inner[0][0][0])
    bot_right_2d_y = int(bot_right_2d_inner[0][0][1])
    # new_img[bot_right_2d_y,bot_right_2d_x, :] = img[750,750,:]

    bot_left_2d_x = int(bot_left_2d_inner[0][0][0])
    bot_left_2d_y = int(bot_left_2d_inner[0][0][1])
    # new_img[bot_left_2d_y,bot_left_2d_x, :] = img[750,650,:]

    # tl_right_outer_x = int(top_left_2d_outer[0][0][0])

    # print('test')
    # print(tl_right_outer_x)
    # new_img[0,0] = img[0,0]
    # new_img[0,width-1] = img[0,width-1]
    # new_img[height-1,0] = img[height-1,0]
    # new_img[height-1, width-1] = img[height-1, width-1]

    # x_pts = [0,top_left_2d_x, width-1, top_right_2d_x]
    # y_pts = [0,top_left_2d_y, 0,top_right_2d_y]

    # MISNAMED X AND Y
    plt.plot((0,top_left_2d_x),(0,top_left_2d_y), marker='o')
    # print(top_left_2d_x, top_left_2d_y)
    plt.plot((width-1,top_right_2d_x),(0,top_right_2d_y), marker='o')
    # print(top_right_2d_x, top_right_2d_y)
    plt.plot((0,bot_left_2d_x),(height-1,bot_left_2d_y), marker='o')
    # print(bot_right_2d_x, bot_right_2d_y)
    plt.plot((width-1,bot_right_2d_x),(height-1,bot_right_2d_y), marker='o')
    # print(bot_left_2d_x, bot_left_2d_y)

    projected_2d_outer = [[0,0],[0,width-1],[height-1, width-1],[height-1, 0]]
    # projected_2d_outer = [ , int(top_left_2d_outer[0][0][0])), int((top_right_2d_outer[0][0][1]), int(top_right_2d_outer[0][0][0])), (int(bot_right_2d_outer[0][0][1]), int(bot_right_2d_outer[0][0][0])), (int(bot_left_2d_outer[0][0][1]), int(bot_left_2d_outer[0][0][0]))]
    projected_2d_inner= [(top_left_2d_y, top_left_2d_x), (top_right_2d_y, top_right_2d_x), (bot_right_2d_y, bot_right_2d_x), (bot_left_2d_y, bot_left_2d_x)]
    print(projected_2d_inner)
    return projected_2d_inner, projected_2d_outer,

    # NOW SET THE TOP LEFT OF NEW TO BE TOP OF LEFT OF OLD AND ON AND ON
    # ONCE YOU HAVE 8 POINTS, TOP, BOTTOM, INNER, OUTER, LEFT, RIGHT
    # DRAW LINES FROM EACH OF THEM TO BUILD THE "edges" of the cube through Z
    # MAKE SURE THAT IT ROUGHLY LOOKS LIKE WHERE WE THINK THE BOX AND LINES TO BE

    #A B C D E F G H
    # old_pts = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1],[625,353],[775,353],[775,533],[625,533]])

  
    


    plt.imshow(final_fill)
    plt.show()


    # print(top_right_2d_x,top_right_2d_y)
    # x_2d = x_p / z_p
    # y_2d = y_p / z_p

    # print(x_2d,y_2d)
    # print(int(x_2d[0]))

    # new_img[]
    # new_img[int(y_2d[0]),int(x_2d[0]),:] = img[0,width-1,:]
    # plt.imshow(new_img)
    # plt.imshow(new_img)
    # plt.show()
    
    # print(points_2d)


    # point correspondances
    # A = [[x1,y1],[x2,y2],...]
    # B = [[x1',y1']...]

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

    # TL LINE (0,625) (0,353)
    # plt.plot((tl_out[1],tl_in[1]), (tl_out[0],tl_in[0]), marker='o')
    # # TR LINE (1399,775) (0,353)
    # plt.plot((tr_out[1],tr_in[1]), (tr_out[0],tr_in[0]), marker='o')
    # # BR LINE (1399,775) (886, 533)
    # plt.plot((br_out[1],br_in[1]), (br_out[0],br_in[0]), marker='o')
    # # BL LINE (0,625) (886, 533)
    # plt.plot((bl_out[1],bl_in[1]), (bl_out[0],bl_in[0]), marker='o')

    #array w inner rect panel
    # inner_rect_mask =(x < inner_rect_pts[2][0]) & (x > inner_rect_pts[0][0]) &(y < inner_rect_pts[2][1]) & (y > inner_rect_pts[0][1])
    # inner_rect_mask = np.stack([inner_rect_mask] * img.shape[2], axis=-1).astype(np.uint8)
    # inner_rect = inner_rect_mask * img
    # plt.imshow(inner_rect)     

    inner_rect_mask = (x >= tl_in[1]) & (x <= tr_in[1]) & (y >= tl_in[0]) & (y <= bl_in[0])
    inner_rect_mask = np.stack([inner_rect_mask] * img.shape[2], axis=-1).astype(np.uint8)
    inner_rect = inner_rect_mask * img

    # TODO: ADD THEM but MAYBE DO THAT AT END BUT ACTUALLY JUST DO THAT TO CHECK

    # left panel : CORRECT
    top_m = (tl_in[0]-tl_out[0])/(tl_in[1]-tl_out[1])
    bot_m = (bl_in[0]-bl_out[0])/(bl_in[1]-bl_out[1])
    left_panel_mask = (x>0) & (x<tl_in[1]) & (y > top_m*x) & (y < (bot_m*x) +bl_out[0])
    left_panel_mask = np.stack([left_panel_mask] * img.shape[2], axis=-1).astype(np.uint8)
    left_rect = (left_panel_mask * img)
    # plt.imshow(left_rect)

    # top panel : CORRECT
    left_m = (tl_in[0]-tl_out[0])/(tl_in[1]-tl_out[1])
    left_intercept = tl_in[0] - (left_m * tl_in[1])
    right_m = (tr_in[0]-tr_out[0])/(tr_in[1]-tr_out[1])
    right_intercept = tr_in[0] - (right_m * tr_in[1])
    top_panel_mask = (y>tl_out[0]) & (y<tl_in[0]) & (y <(left_m*x) + left_intercept) & (y < (right_m*x) + right_intercept)
    top_panel_mask = np.stack([top_panel_mask] * img.shape[2], axis=-1).astype(np.uint8)
    top_rect = (top_panel_mask * img)
    # plt.imshow(top_rect)

    # right panel : CORRECT
    top_m = (tr_in[0]-tr_out[0])/(tr_in[1]-tr_out[1])
    top_intercept = tr_in[0] - (top_m * tr_in[1])
    bot_m = (br_in[0]-br_out[0])/(br_in[1]-br_out[1])
    bot_intercept = br_in[0] - (bot_m * br_in[1])
    right_panel_mask = (x>tr_in[1]) & (x<tr_out[1]) & (y > (top_m*x) + top_intercept) & (y < (bot_m*x) + bot_intercept)
    right_panel_mask = np.stack([right_panel_mask] * img.shape[2], axis=-1).astype(np.uint8)
    right_rect = (right_panel_mask * img) 
    # plt.imshow(right_rect)

    # bottom panel: TODO
    left_m = (bl_in[0]-bl_out[0])/(bl_in[1]-bl_out[1])
    left_intercept = bl_in[0] - (left_m * bl_in[1])
    right_m = (br_in[0]-br_out[0])/(br_in[1]-br_out[1])
    right_intercept = br_in[0] - (right_m * br_in[1])
    bottom_panel_mask = (y>bl_in[0]) & (y<bl_out[0]) & (y > (left_m*x) + left_intercept) & (y > (right_m*x) + right_intercept)
    bottom_panel_mask = np.stack([bottom_panel_mask] * img.shape[2], axis=-1).astype(np.uint8)
    bottom_rect = (bottom_panel_mask * img)
    plt.imshow(bottom_rect)

    new = np.zeros_like(img)
    new+= inner_rect+left_rect + top_rect + right_rect + bottom_rect

    # able to cut into each
    plt.imshow(new)
    plt.show()

    return inner_rect, left_rect, top_rect, right_rect, bottom_rect

def createHomography(old_quad, new_quad, img):
    # old_inner_rect = np.array([[625,353],[775,353],[775,533],[625,533]])
    # new_inner_rect = np.array([[top_left_2d_x,top_left_2d_y],[top_right_2d_x,top_right_2d_y],[bot_right_2d_x,bot_right_2d_y],[bot_left_2d_x,bot_left_2d_y]])
    # # are first four always going to be the same ????????????
    # # new_pts = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1],[top_left_2d_x,top_left_2d_y],[top_right_2d_x,top_right_2d_y],[bot_right_2d_x,bot_right_2d_y],[bot_left_2d_x,bot_left_2d_y]])
    # # new_pts = 
    
    M,_ = cv2.findHomography(old_quad,new_quad)
    # M, _ = cv2.findHomography(old_inner_rect,new_inner_rect)

    out = cv2.warpPerspective(img,M,(width,height))
    # old_img_inner_rect = img[353:533, 625:775, :]

    # output = cv2.warpPerspective(old_img_inner_rect, M,(width,height))
    
    final_fill = np.zeros((height, width, 3))
    # np.where(out != 0, out, final_fill)
    final_fill+= out
    plt.imshow(final_fill)
    plt.show()

    # final_fill[top_left_2d_y:bot_left_2d_y, top_left_2d_x:top_right_2d_x, :] = old_img_inner_rect


if __name__ == '__main__':
    # full image
    # background image
    # foreground image
    img = cv2.imread("data/1_full.jpg")
    # background = cv2.imread("data/1_background.jpg")
    # foreground = cv2.imread("data/1_foreground.jpg")

    img = img.astype(np.float32) / 255.
    # background = background.astype(np.float32) / 255.
    # foreground = foreground.astype(np.float32) / 255.

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    # foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    
    # width = img.shape[1]
    # # print(width)
    # height = img.shape[0]
    # # print(height)
    plt.imshow(img)
    plt.show()

    height, width, _ = img.shape
    # inner_rect_pts = [[625,353],[625,533],[775,533],[775,353]]
    # where the first in the tuple is how far down it is and the second is how for to the right it is
    inner_rect_pts = [[353,625],[353,775],[533,775],[533,625]]
    # inner_rect_pts = [[533, 625], [533, 775], [713, 775], [713, 625]]
    # inner_rect_pts = [[533, 775], [533, 925], [713, 925], [713, 775]]
    outer_rect_pts =[[0,0],[0,width-1],[height-1, width-1],[height-1, 0]]

    inner,left,top,right,bot = create_side_images(img,inner_rect_pts, outer_rect_pts, width, height)

    projected_inner,project_outer = project3Dto2D(img)

    old_left = np.array([outer_rect_pts[0],inner_rect_pts[0],inner_rect_pts[3],outer_rect_pts[3]])
    new_left = np.array([project_outer[0],projected_inner[0],projected_inner[3],project_outer[3]])
    createHomography(old_left,new_left,left)
    


    # print(projected_inner,project_outer)
    # plt.imshow(background)
    # plt.show()
    # plt.imshow(foreground)
    # plt.show()