import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import patches,  lines



def TwoD2ThreeD(P, coord, z):
    # transform 2D points in the image to 3D points in the space
	# input:    P: (3, 4) projection matrix
	#           coord: 2D coordinate matrix (non-homo) of size (2, N)
	#           z: array of z in 3D space of size (N,)
	# output:   3D coordinate matrix (non-homo) of size (3, N) 
	temp_P = np.copy(P[:, 0:3]).astype(float)
	TwoD_homo = convertToHomo(coord)
	res = np.zeros((3, coord.shape[1]))
	res[2, :] = z
	
	for i in range(coord.shape[1]):
		temp_P[:, 2] = z[i]*P[:, 2] + P[:, 3]
		temp = np.dot(np.linalg.inv(temp_P), TwoD_homo[:, i])
		res[0, i] = temp[0]/temp[2]
		res[1, i] = temp[1]/temp[2]
    
	return res


def ThreeD2TwoD(P, coord):
    # transform 3D points in the space to 2D points in the image
    # input:    P: (3, 4) projection matrix
    #           coord: 3D coordinate matrix (non-homo) of size (3, N)
    # output:   2D coordinate matrix (non-homo) of size (2, N)
    ThreeD_homo = convertToHomo(coord)
    TwoD_homo = P.dot(ThreeD_homo)
    return convertFromHomo(TwoD_homo)

def convertToHomo(coord):
    res = np.ones((coord.shape[0]+1, coord.shape[1]))
    res[:coord.shape[0], :] = coord
    return res

def convertFromHomo(coord):
    res = coord[:-1, :]
    for i in range(res.shape[0]):
        res[i, :] = res[i, :]/coord[-1,:]
    return res


def load_gt_disp_kitti(path):
    gt_disparities = []
    disp = cv2.imread(path, -1)
    disp = disp.astype(np.float32) / 256
    gt_disparities.append(disp)
    return gt_disparities


def convert_disps_to_depths_kitti(gt_disparities, pred_disparities, width_to_focal):
    gt_depths = []
    pred_depths = []
    pred_disparities_resized = []

    
    for i in range(len(gt_disparities)):
        gt_disp = gt_disparities[i]
        height, width = gt_disp.shape

        pred_disp = pred_disparities[i]
        pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)

        pred_disparities_resized.append(pred_disp) 
        # print(gt_disp)
        mask = gt_disp > 0

        # 0.54, 0.5707 width_to_focal[width]
        gt_depth =  width_to_focal[width] *  0.54 / (gt_disp + (1.0 - mask))
        pred_depth = width_to_focal[width] * 0.54 / pred_disp

        gt_depths.append(gt_depth)
        pred_depths.append(pred_depth)
    return gt_depths, pred_depths, pred_disparities_resized






def plot_mask(image, masks):
    # only plot 2D region of the object detected by Mask-RCNN
    height, width = image.shape[:2]
    trial_img = np.zeros((height, width, 3))
    _, ax = plt.subplots(1)
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    # ax.set_title(title)
    alpha = 0.5
    N = masks.shape[2]
    for i in range(N):
        mask = masks[:, :, i]
        mask_pos = np.where((mask==1))
        # print("0 (y) range {}~{}, 1 (x) range: {}~{}".format(np.min(mask_pos[0]), np.max(mask_pos[0]), 
        #                                 np.min(mask_pos[1]), np.max(mask_pos[1])))
        for c in range(3):
            trial_img[:, :, c] = np.where(mask == 1,
                                    trial_img[:, :, c] *
                                    (1 - alpha) + alpha * 1 * 255,
                                    trial_img[:, :, c])
    
    ax.imshow(trial_img.astype(np.uint8))
    plt.show()

def get_pcd_masked(masks, depth_data, P, is_show = False, threshold = 500):
    # only plot 3D points from the object detected by Mask-RCNN
    # input:    mask: a list of output masks from MaskRCNN with size of (H,W,N)
    #           depth_data: depth information for the same image
    #           P: (3, 4) projection matrix
    # output:   pcd_3D_list: list of array of object position in 3D space with size of (3, N)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pcd_3D_list = []
    for idx in range(masks.shape[2]):
        mask = masks[:,:,idx]
        mask_pos = np.where((mask==1)*(depth_data < 200.0))  #
        # print("mask size = {}".format(mask_pos[0].shape))
        # print("filered size 1: {}".format(mask_pos[0].shape[0]))
        if(mask_pos[0].shape[0] < threshold):
            continue

        depth_masked = depth_data[mask_pos]

        d_median = np.median(depth_masked)
        d_std = np.std(depth_masked)
        inlier_idx = np.where(np.abs(depth_masked - d_median) < 3)
        # print("filered size 2: {}".format(inlier_idx[0].shape[0]))
        if(inlier_idx[0].shape[0] < threshold):
            continue
        depth_masked = depth_masked[inlier_idx]


        pcd_2D = np.array([mask_pos[1][inlier_idx], mask_pos[0][inlier_idx]])

        # print("After mask: xy: {}, z: {}".format(pcd_2D.shape, depth_masked.shape))
        pcd_3D = TwoD2ThreeD(P, pcd_2D, depth_masked)
        pcd_3D_list.append(pcd_3D)
        if(is_show):
            ax.scatter(pcd_3D[0, :], pcd_3D[1, :], pcd_3D[2, :], c='r', marker='.')
            vertice3D, edge = boundingbox(pcd_3D)
            for i in range(edge.shape[0]):
                #line_plot = np.array([],[])
                ax.plot([vertice3D[0, edge[i, 0]], vertice3D[0, edge[i, 1]]],
                    [vertice3D[1, edge[i, 0]], vertice3D[1, edge[i, 1]]],
                    [vertice3D[2, edge[i, 0]], vertice3D[2, edge[i, 1]]], 'b')
    print("{} mask, {} saved".format(masks.shape[2], len(pcd_3D_list)))
    
    
    
    if(is_show):
        Xmax, Xmin, Ymax, Ymin, Zmax, Zmin = 10, -10, 10, 0, 60, 0
        ax.plot([0, 0],[0, 0], [Zmax, 0])
        ax.axes.set_xlim3d(left=-20, right=20)
        ax.axes.set_ylim3d(bottom=-2, top=6) 
        ax.axes.set_zlim3d(bottom=0, top=60) 
        
        ax.view_init(elev = -40, azim = -90)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        plt.grid()
        plt.show()
    return pcd_3D_list

def boundingbox(pcd_3D):
    # generate 3D bounding box
    xyz_max = np.max(pcd_3D, axis=1)
    xyz_min = np.min(pcd_3D, axis=1)
    x1, x2 = xyz_min[0],xyz_max[0]
    y1, y2 = xyz_min[1],xyz_max[1]
    z1, z2 = xyz_min[2],xyz_max[2] + 2 
    vertice3D = np.array([[x1, x1, x1, x1, x2, x2, x2, x2],
                        [y1, y1, y2, y2, y1, y1, y2, y2],
                        [z1, z2, z1, z2, z1, z2, z1, z2]])
    edge = np.array([[0, 1],[0, 2],[2, 3],[1, 3],[0, 4],[1, 5],
                        [2, 6],[3, 7], [5, 7],[4, 5],[6, 7],[4, 6] ])
    return vertice3D, edge

def get_boundingbox(pcd_3D_list, image, P, res_dir, img_name):
    # plot the bounding box back to 2D orginal image
    # input: pcd_3D_list a list of 3D coordinate of detected objects, with size of (3, N)
    #       image: origin 2D image
    #       P: transformation matrix of size (3, 4)
    # output: None
    fig, ax1 = plt.subplots(ncols=1)
    ax1.imshow(image)
    for idx in range(len(pcd_3D_list)):
        pcd_3D = pcd_3D_list[idx]
        vertice3D, edge = boundingbox(pcd_3D)
        vertice2D = ThreeD2TwoD(P, vertice3D)
        for i in range(edge.shape[0]):
            ax1.plot([vertice2D[0, edge[i, 0]], vertice2D[0, edge[i, 1]]],
                [vertice2D[1, edge[i, 0]], vertice2D[1, edge[i, 1]]], 'r')
    
    # plt.show()
    ax1.axis('off')
    plt.tight_layout()
    plt.savefig(res_dir+img_name+'_box.png', bbox_inches='tight', pad_inches=0, dpi=199)    #
    

####################### NOT USED BELOW #######################

def generate_xy(d_dim):
    # generate all 2D coordinate in an image of size d_dim
    # output: (2, d_dim[0]*d_dim[1]) array
	Y_size, X_size = d_dim
	coord_y = np.repeat(np.array(range(Y_size)), X_size).astype(float)	# column direction
	coord_x = np.repeat(np.array(range(X_size)), Y_size).astype(float)
	coord_x = (coord_x.reshape(-1, Y_size)).transpose().reshape(-1)		# row direction
	coord_x = np.expand_dims(coord_x, axis=0)
	coord_y = np.expand_dims(coord_y, axis=0)
	return np.concatenate((coord_x, coord_y), axis=0)

def plot_pcd(depth_data, P):
    # plot all 3D points from depth image
    ref_pcd_2D = generate_xy(depth_data[0].shape)
    ref_pcd_3D = TwoD2ThreeD(P, ref_pcd_2D, depth_data[0].reshape(-1))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ref_pcd_3D[0, :], ref_pcd_3D[1, :], ref_pcd_3D[2, :], c='r', marker='.')
    plt.show()


def test_transformation():
    a = np.array([[1, 1, 1], [2.15, 2.15, 2.15], [1, 4, 2], [1, 1, 0.5]])
    a = a.T
    f = 2
    Pin = np.array([[f, 0 ,2],[0, f , 2],[0, 0, 1]]).astype(float)
    Pex = np.array([[1, 0 ,2, 0],[0, 1 , 2, 0],[0, 0, 1, 1]]).astype(float)
    P = Pin.dot(Pex)
    P = np.array([[0, -1, 0, 0],[1, 0, 0, 0],[0, 0, 1, 0]]).astype(float)
    print("original 3D: ")
    print(a)
    test_res = ThreeD2TwoD(P, a)
    print("converted 2D:")
    print(test_res)
    test_a = TwoD2ThreeD(P, test_res, a[2, :])
    print("calculated 3D:")
    print(test_a)

