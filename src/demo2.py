import os
import sys
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from boundingbox_helper import *
from mask_rcnn_handler import *
from monodepth_handler import *
from calibration_handler import *
from d2z_handler import *
import numpy as np
import skimage.io



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
    D2Z_obj = d2zClass()
    for img_name in img_name_list:
        cur_image = skimage.io.imread(left_dir+img_name+'.png')
        cur_image_name = left_dir+img_name+'.png'
        # cur_gt_path = disp_dir+img_name+'.png'
        # cur_gt_disparities = load_gt_disp_kitti(cur_gt_path)
        original_shape = [cur_image.shape[0], cur_image.shape[1]]
        # print(original_shape)   # 375, 1242

        if(mode_pred == 'complete'):
            # predict disparity and mask
            cur_masks= MaskRCNN_obj.predict(cur_image_name, is_savep = False, is_show = False)
            cur_pred_disparities = [Monodepth_obj.predict(cur_image_name)]
        else:
            # load pre-calculated disparity and mask
            cur_masks = pickle.load( open(res_dir+img_name+'.p', "rb" ))
            cur_pred_disparities = [np.load(res_dir+img_name+'_disp.npy')]

        
        cur_P_rect = Calibration_obj.getP(img_name)
        # 
        if(mode_d2z == 'classical'):
            pred_depths = \
                convert_disps_to_depths_kitti(cur_pred_disparities, Calibration_obj.width_to_focal, original_shape, mode = 'pred')  # shape (375,1242)
            # gt_depths, pred_depths, pred_disparities_resized = \
            #     convert_disps_to_depths_kitti(cur_gt_disparities, cur_pred_disparities, Calibration_obj.width_to_focal)  # shape (375,1242)
        else:
            # load trained z directly:
            # pred_depths = [np.load(res_dir + img_name+'_simple.npy')]
            pred_depths = D2Z_obj.predict(cur_pred_disparities, original_shape)
            

        # print(cur_masks['rois']) #y1, x1, y2, x2 
        cur_pcd_3D_list = get_pcd_masked(cur_masks, pred_depths[0], cur_P_rect, False, 500)
        # cur_pcd_3D_list = get_pcd_masked(cur_masks, gt_depths[0], cur_P_rect)

        get_boundingbox(cur_pcd_3D_list, cur_image, cur_P_rect, res_dir, img_name, z_offset = 0)  


if __name__ == '__main__':
    # how to run:
    #   python demo2.py 000056_10   # handle the img 000056_10.png
    #   python demo2.py all         # handle the default img (000056_10.png)  
    #   python demo2.py             # handle all images in left_dir

    mode_d2z = 'classical'     # mode_d2z = 'classical' or 'nn'
    mode_pred = 'complete'  # mode_pred = 'pre-processed' or 'complete'
    # TODO: dir name to be changed
    left_dir = '../images/data/left_img/'
    right_dir = '../images/data/right_img/'
    disp_dir = '../images/data/disp_img/'   # optional
    calibration_dir = '../images/data/calibration/'

    res_dir = '../images/res/'
    main()