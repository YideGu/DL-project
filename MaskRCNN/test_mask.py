import os
import sys
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from reconstruct import *
import numpy as np
import skimage.io

from mrcnn import utils
# import mrcnn.model as modellib
# from mrcnn import visualize
# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# import coco

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351
P_rect = np.array([[721.5377, 0.000000, 609.5593, 44.85728],
    [0.000000, 721.5377, 172.8540, 0.2163791],
    [0.000000, 0.000000, 1.000000, 0.002745884]])





image = skimage.io.imread("../images/000056_10l.png")
r = pickle.load( open( "MaskRCNN_res.p", "rb" ) )
predicted_disp_path = "../images/000056_10l_disp.npy"   # the output of monodepth repo
gt_path = "../images/depth/000056_10dl.png"
pred_disparities = [np.load(predicted_disp_path)]
gt_disparities = load_gt_disp_kitti(gt_path)
gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_kitti(gt_disparities, pred_disparities)  # shape (375,1242)

# print("image {}, roi {}, masks {}".format(image.shape, r['rois'].shape, 
#     r['masks'].shape))

N = r['masks'].shape[2]


masks = r['masks']
pcd_3D_list = get_pcd_masked(masks, pred_depths[0], P_rect)
get_boundingbox(pcd_3D_list, image, P_rect)  

# print("TEST: gt depth {}, pred depth {}".format(gt_depths[0].shape, pred_depths[0].shape))
# plot_pcd(gt_depths)
# plot_pcd(pred_depths)