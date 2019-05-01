from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import sys


import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import skimage.transform

ROOT_DIR =  "../disparity2z/"   #os.path.abspath( "../")
# Import d2z
sys.path.append(ROOT_DIR)  
from simplecnn import *

class d2zClass:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        MODEL_FILENAME = '../../model/d2z_model/'+"trained_model_cnn.pth"
        self.net = simplecnn().to(self.device)
        self.net.load_state_dict(torch.load(MODEL_FILENAME))

    def predict(self, disp_map_list, original_shape = (375, 1242), filename = '../images/data/left_img/000056_10.png', res_dir = '../images/res/'):
        dpt_list = []
        for disp_map in  disp_map_list:
            disp_map = torch.tensor(disp_map, dtype=torch.float32).to(self.device)
            # h, w = disp_map.shape
            print(disp_map.shape)
            disp_to_depth = self.net(disp_map.resize(1,1,256, 512)).reshape(256, 512)
            dpt = disp_to_depth.to('cpu').detach().numpy()
            k = np.max(np.abs(dpt))
            dpt = dpt / k
            dpt = cv2.resize(dpt, (original_shape[1], original_shape[0]))
            dpt = dpt * k
            dpt_list.append(dpt)
            output_name = (filename.split('/')[-1]).split('.')[0]
            plt.imsave(os.path.join(res_dir+output_name+'_depth_cnn.png'), dpt, cmap='plasma')

        return dpt_list
