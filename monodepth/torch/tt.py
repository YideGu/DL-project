import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt

'''
tmp = np.load('data/output/kitti/test/180426/2/proj_cloud_point.npy')
print(tmp.shape)
for i in range(len(tmp)):
	for j in range(512):
		kk = tmp[i, :, j].copy()
		for k in range(256):
			tmp[i, k, j] = kk[255 - k]
np.save('data/output/kitti/test/180426/2/proj_cloud_point.npy', tmp)
'''
dispn = np.load('data/output/kitti/test/180426/2/disp.npy')[0].reshape(256, 512)
pj = np.load('data/output/kitti/test/180426/2/proj_cloud_point.npy')[0].reshape(256, 512)
dp0 = np.load('data/output/kitti/test/180426/3/depth0.npy')
dp1 = np.load('data/output/kitti/test/180426/2/depth0.npy')
loss1 = np.load('data/output/kitti/test/180426/3/loss.npy')
loss2 = np.load('data/output/kitti/test/180426/2/loss.npy')[40:]

fig1 = plt.figure(1, figsize=(32, 4))
ax1 = fig1.add_subplot(1, 2, 1)
#p = ax1.imshow(dp0, cmap='jet')
p = ax1.semilogy(0.1 * loss1[::5])
plt.xlim([0,200])
plt.xlabel('epoch')
#fig1.colorbar(p, orientation='vertical')

ax2 = fig1.add_subplot(1, 2, 2)
pp = ax2.semilogy(0.1 * loss2[::5])
plt.xlim([0,200])
plt.xlabel('epoch')
#pp = ax2.imshow(dp1, cmap='jet')
#fig1.colorbar(pp, orientation='vertical')
plt.show()
plt.close()


