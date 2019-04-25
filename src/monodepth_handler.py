# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt
import sys

# TODO: ROOT_DIR to be changed
# Root directory of the project
ROOT_DIR_depth =  "../monodepth"  
sys.path.append(ROOT_DIR_depth)  # To find local version of the library
#from __future__ import absolute_import, division, print_function

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *

class MonodepthPredClass:
    def __init__(self, in_input_height=256, in_input_width=512, in_encoder = 'vgg', \
                checkpoint_path = '../../model/monodepth_model/model_kitti'):
        self.input_height = in_input_height
        self.input_width = in_input_width
        self.left  = tf.placeholder(tf.float32, [2, self.input_height, self.input_width, 3])

        self.params = monodepth_parameters(
            encoder=in_encoder,
            height=self.input_height,
            width=self.input_width,
            batch_size=2,
            num_threads=1,
            num_epochs=1,
            do_stereo=False,
            wrap_mode="border",
            use_deconv=False,
            alpha_image_loss=0,
            disp_gradient_loss_weight=0,
            lr_loss_weight=0,
            full_summary=False)

        # define model
        self.model = MonodepthModel(self.params, "test", self.left, None)

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)

        # SAVER
        self.train_saver = tf.train.Saver()

        # INIT
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=coordinator)

        # RESTORE
        restore_path = checkpoint_path.split(" ")[0]
        print('load model from '+restore_path)
        self.train_saver.restore(self.sess, restore_path)

    def predict(self, filename = '../images/data/left_img/000056_10.png', is_saven = False, res_dir = '../images/res/'):
        print("predict depth for {}".format(filename))
        input_image = scipy.misc.imread(filename, mode="RGB")
        original_height, original_width, num_channels = input_image.shape
        input_image = scipy.misc.imresize(input_image, [self.input_height, self.input_width], interp='lanczos')
        input_image = input_image.astype(np.float32) / 255
        input_images = np.stack((input_image, np.fliplr(input_image)), 0)

        # get result
        disp = self.sess.run(self.model.disp_left_est[0], feed_dict={self.left: input_images})
        disp_pp = self.post_process_disparity(disp.squeeze()).astype(np.float32)

        output_name = (filename.split('/')[-1]).split('.')[0]
        if(is_saven):
            np.save(res_dir+output_name+'_disp.npy', disp_pp)
        disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
        plt.imsave(os.path.join(res_dir+output_name+'_disp.png'), disp_to_img, cmap='plasma')

        print('done!')
        return disp_pp
    
    def post_process_disparity(self, disp):
        _, h, w = disp.shape
        l_disp = disp[0,:,:]
        r_disp = np.fliplr(disp[1,:,:])
        m_disp = 0.5 * (l_disp + r_disp)
        l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
        r_mask = np.fliplr(l_mask)
        return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp