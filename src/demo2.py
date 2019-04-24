import os
import sys
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from boundingbox_helper import *
from mask_rcnn_handler import *
from monodepth_handler import *
from calibration_handler import *
import numpy as np
import skimage.io

# from mrcnn import utils
# import mrcnn.model as modellib
# from mrcnn import visualize
# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# import coco



def handle_input():
    # get the list of image name to handle
    img_name_list = []
    if(len(sys.argv) < 2):
        # default img to process
        img_name_list.append('000056_10')
    elif(sys.argv[1] == 'all'):
        for (_,_,files) in os.walk(left_dir, topdown=True):
            for idx in range(0, len(files)):
                #print(files[i].split('.')[0])
                img_name_list.append(files[idx].split('.')[0])
    else:
        for idx in range(1, len(sys.argv)):
            img_name_list.append(sys.argv[idx])
    return img_name_list

def main():
    img_name_list = handle_input()
    
    Monodepth_obj = MonodepthPredClass()
    MaskRCNN_obj = MaskPredictClass()
    Calibration_obj = CalibrationClass()
    for img_name in img_name_list:
        cur_image = skimage.io.imread(left_dir+img_name+'.png')
        cur_gt_path = disp_dir+img_name+'.png'
        cur_gt_disparities = load_gt_disp_kitti(cur_gt_path)

        # predict disparity and mask
        # cur_maskrcnn_res = MaskRCNN_obj.predict(is_savep = False, is_show = False)
        Monodepth_obj.predict()
        # load disparity and mask
        cur_maskrcnn_res = pickle.load( open(res_dir+img_name+'.p', "rb" ))
        cur_pred_disparities = [np.load(res_dir+img_name+'_disp.npy')]

        # print("image {}, roi {}, masks {}".format(cur_image.shape, r['rois'].shape, r['masks'].shape))
        # N = cur_maskrcnn_res['masks'].shape[2]

        cur_P_rect = Calibration_obj.getP(img_name)

        gt_depths, pred_depths, pred_disparities_resized = \
            convert_disps_to_depths_kitti(cur_gt_disparities, cur_pred_disparities, Calibration_obj.width_to_focal)  # shape (375,1242)
        
        cur_masks = cur_maskrcnn_res['masks']
        cur_pcd_3D_list = get_pcd_masked(cur_masks, pred_depths[0], cur_P_rect)
        get_boundingbox(cur_pcd_3D_list, cur_image, cur_P_rect)  

        # print("TEST: gt depth {}, pred depth {}".format(gt_depths[0].shape, pred_depths[0].shape))
        # plot_pcd(gt_depths)
        # plot_pcd(pred_depths)

if __name__ == '__main__':
    # how to run:
    #   python test_mask.py 000056_10   # handle the img 000056_10.png
    #   python test_mask.py all         # handle the default img (000056_10.png)  
    #   python test_mask.py             # handle all images in left_dir

    # TODO: dir name to be changed
    left_dir = '../images/data/left_img/'
    right_dir = '../images/data/right_img/'
    disp_dir = '../images/data/disp_img/'
    calibration_dir = '../images/data/calibration/'
    res_dir = '../images/res/'
    main()