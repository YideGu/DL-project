import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
import os
import re

# read
imgLset = sorted(glob.glob('data/kitti/test/*/image_02/data/*'))
imgRset = sorted(glob.glob('data/kitti/test/*/image_03/data/*'))
imgDset = sorted(glob.glob('data/output/kitti/test/pred_*'), key=lambda x:float(re.findall("(\d+)",x)[0]))
idx = np.random.choice(len(imgLset))
idx = 28
imgL = plt.imread(imgLset[idx],1)
imgR = plt.imread(imgRset[idx],1)
imgD = plt.imread(imgDset[idx],1)


# plot
f = plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
plt.imshow(imgL)
#plt.subplot(3,1,2)
#plt.imshow(imgR)
plt.subplot(2,1,2)
plt.imshow(imgD)
plt.show()
plt.close()