
# coding: utf-8

import os
import torch
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

from main_monodepth_pytorch import Model


#torch.cuda.is_available()
#torch.cuda.device_count()

torch.cuda.empty_cache()

dicname = str(3)
'''
epochs = 20
dict_parameters = edict({'data_dir':'data/kitti/train/',
                         'val_data_dir':'data/kitti/test/',
                         'model_path':'data/models/monodepth_resnet18_002_last.pth',
                         'output_directory':'data/output/',
                         'input_height':256,
                         'input_width':512,
                         'model':'resnet18_md',
                         'pretrained':True,
                         'mode':'train',
                         'epochs':20,
                         'learning_rate':1e-4,
                         'batch_size': 2,
                         'adjust_lr':True,
                         'device':'cuda:0',
                         'do_augmentation':True,
                         'augment_parameters':[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                         'print_images':False,
                         'print_weights':False,
                         'input_channels': 3,
                         'num_workers': 8,
                         'use_multiple_gpu': False})


model = Model(dict_parameters)
model.load('data/models/monodepth_resnet18_002_cpt.pth')

#model.train()
'''
# Testing
dicdate = '180424/'
os.system('mkdir data/output/kitti/test/'+dicdate+dicname)


dict_parameters_test = edict({'data_dir':'data/kitti/test/test/',
                              'model_path':'data/models/monodepth_resnet18_002.pth',
                              'output_directory':'data/output/kitti/test/'+dicdate+dicname+'/',
                              'input_height':256,
                              'input_width':512,
                              'model':'resnet18_md',
                              'pretrained':False,
                              'mode':'test',
                              'device':'cuda:0',
                              'input_channels':3,
                              'num_workers':4,
                              'use_multiple_gpu':False})
model_test = Model(dict_parameters_test)

model_test.test()

disp = np.load('data/output/kitti/test/'+dicdate+dicname+'/disparities_pp.npy')  # Or disparities.npy for output without post-processing

disp_to_img = skimage.transform.resize(disp[0].squeeze(), [375, 1242], mode='constant')
plt.imshow(disp_to_img, cmap='plasma')


# Save images

plt.imsave(os.path.join(dict_parameters_test.output_directory,
                        dict_parameters_test.model_path.split('/')[-1][:-4]+'_test_output.png'), disp_to_img, cmap='plasma')

for i in range(disp.shape[0]):
    disp_to_img = skimage.transform.resize(disp[i].squeeze(), [375, 1242], mode='constant')
    plt.imsave(os.path.join(dict_parameters_test.output_directory,
               'pred_'+str(i)+'.png'), disp_to_img, cmap='plasma')

#plt.imsave(os.path.join(dict_parameters_test.output_directory,
#                        dict_parameters_test.model_path.split('/')[-1][:-4]+'_gray.png'), disp_to_img, cmap='gray')

