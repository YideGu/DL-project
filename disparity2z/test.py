#! /usr/bin/python3
# from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
# import pykitti
# from itertools import compress
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from simple import *
import skimage.transform




"""
projlist = np.zeros((trainlen, 256, 512)); projlist[:, :, :] = np.nan

for frame in range(trainlen):
	veloplot = data_velo[frame].T
	xyz = veloplot[:3, :]
	xyz[2, :] = -xyz[2, :]
	arg = []
	for i in range(len(xyz[0])):
		if(xyz[0][i] > 0):
			arg.append(i)
	xyz = xyz[:, arg]
	clr = np.linalg.norm(xyz, axis=0)
	uv = -proj2 @ np.vstack([xyz, np.ones_like(xyz[0, :])])
	uv = uv / uv[2, :]
	arg = []
	for i in range(len(uv[0])):
		if(uv[0][i] <= 1242 and uv[0][i] >= 0 and uv[1][i] <= 375 and uv[1][i] >= 0):
			arg.append(i)
	uv = uv[:, arg]
	xyz = xyz[:, arg]

	'''
	clr2 = np.floor(64*5/xyz[0, :])

	img = dataset.get_cam2(frame)
	img = np.array(list(img.getdata()))
	img = np.reshape(img, (height, width, 3))
	img2 = img[::-1, :, :]
	step = 1
	
	# figure  2d
	fig1 = plt.figure(1, figsize=(8, 16))
	ax1 = fig1.add_subplot(1, 1, 1)
	#ax1.imshow(img2)
	p = ax1.scatter(uv[0, ::step], uv[1, ::step], c=xyz[0][::step], marker='.', s=1, cmap='jet')
	plt.xlim([0, 1242])
	plt.ylim([0, 375])
	fig1.colorbar(p, orientation='vertical')
	ax1.axis('scaled')
	'''
	
	for i in range(len(uv[0])):
		xx = int(uv[0][i] * 512/1242)
		yy = int(uv[1][i] * 256/375)
		projlist[frame, yy, xx] = xyz[0][i]

	'''
	fig2 = plt.figure(2, figsize=(8, 16))
	ax2 = fig2.add_subplot(1, 1, 1)
	pp = ax2.imshow(matf, cmap='jet')
	plt.xlim([0, 512])
	plt.ylim([0, 256])
	fig2.colorbar(pp, orientation='vertical')
	ax2.axis('scaled')
	'''

	# figure 3d
	'''
	fig2 = plt.figure(2, figsize=(8, 8))
	ax2 = Axes3D(fig2)
	ax2.set_xlabel('x')
	ax2.set_ylabel('y')
	ax2.set_zlabel('z')

	ax2.scatter(
	    xyz[0, ::step], xyz[1, ::step], xyz[2, ::step], marker='.', c=clr[::step], s=1)
	ax2.auto_scale_xyz([-80, 40], [-20, 80], [-4, 10])
	'''
	plt.show()
	plt.close()

np.save('data/output/kitti/test/180426/1/proj_cloud_point', projlist)
"""





def train_model():
	loss = []
	for epochi in range(epoch):

		running_loss = 0.0
		batchnum = disp.shape[0] // batchsize

		for i in range(batchnum):

			batch_idx = np.random.choice(trainlen, batchsize, replace=False)
			disp_batch = disp[batch_idx, :].to(device)
			projlist_batch = projlist[batch_idx, :].to(device)

			optimizer.zero_grad()

			predrect_batch = net_simple(disp_batch)
			diff = predrect_batch-projlist_batch

			running_loss = torch.mean(torch.abs(diff[diff == diff])).to(device)
			loss.append(running_loss)

			running_loss.backward()
			optimizer.step()

			if i % 10 == 0:
				print('[epoch %d, iter %3d] \n     - loss: %.8f' %
	                      (epochi + 1, i + 1, running_loss / batchsize))

	np.save(os.path.join(path, 'loss'), loss)
	# save trained model      
	torch.save(net_simple.state_dict(), MODEL_FILENAME)		

def test_model():
	out = []
	for i in range(disp.shape[0]):
		disp_to_depth = net_simple(disp[i,:,:,:]).reshape(256, 512)
		dpt = disp_to_depth.to('cpu').detach().numpy()
		k = np.max(np.abs(dpt))
		dpt = dpt / k
		#print(disp[i].numpy())
		dpt = skimage.transform.resize(dpt, [375, 1242], mode='constant')
		dpt = dpt * k
		if i == 0:
			f = plt.figure(1, figsize=(8,16))
			#print(k)
			p = plt.imshow(dpt)
			f.colorbar(p, orientation='vertical')
			plt.show()
			plt.close()
		out.append(dpt)
	np.save(os.path.join(path, 'depth'), np.array(out))


def visualize_result(pred_data):
	a = np.load(os.path.join(path, pred_data))
	dpt = projlist[0].to('cpu').numpy().reshape((256, 512))
	
	fig1 = plt.figure(1, figsize=(8, 16))
	ax1 = fig1.add_subplot(1, 1, 1)
	p = ax1.imshow(a, cmap='jet')
	plt.xlim([0, 1242])
	plt.ylim([0, 375])
	fig1.colorbar(p, orientation='vertical')
	ax1.axis('scaled')
	plt.show()
	plt.close()

	fig2 = plt.figure(2, figsize=(8, 16))
	ax2 = fig2.add_subplot(1, 1, 1)
	pp = ax2.imshow(dpt, cmap='jet')
	plt.xlim([0, 512])
	plt.ylim([0, 256])
	fig2.colorbar(pp, orientation='vertical')
	ax2.axis('scaled')
	plt.show()
	plt.close()

	fig3 = plt.figure(3, figsize=(8, 16))
	ax3 = fig3.add_subplot(1, 1, 1)
	a = dispn[0,:,:,0].reshape(256,512)
	ppp = ax3.imshow(a, cmap='jet')
	plt.xlim([0, 512])
	plt.ylim([0, 256])
	fig3.colorbar(pp, orientation='vertical')
	ax3.axis('scaled')
	plt.show()
	plt.close()



if __name__ == '__main__':
	mode = 'train'

	# load data:
	height = 375
	width = 1242

	path = '../../data'
	# path = 'data/output/kitti/test/180426/3'
	#basedir = glob('data/kitti/test/2011_09_26_drive_0014_sync/')
	# dataset = pykitti.raw('data/kitti/test', '2011_09_26', '0009')
	# data_velo = list(dataset.velo)
	# calib = dataset._load_calib_cam_to_cam('calib_velo_to_cam.txt', 'calib_cam_to_cam.txt')
	# P = calib['P_rect_20']
	# R = calib['R_rect_00']
	# T = calib['T_cam2_velo']
	# proj2 = P @ R @ T

	trainlen = min(1000, 443)	# len(data_velo)
	projlist = np.load(os.path.join(path,'proj_cloud_point.npy'))[:trainlen]
	dispn = np.load(os.path.join(path,'disp.npy'))[:trainlen].reshape(trainlen, 256, 512, 1)


	# initialize model:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	epoch = 50
	batchsize = min(10, trainlen)
	torch.manual_seed(598)

	disp = torch.tensor(dispn, dtype=torch.float32).to(device)
	projlist = torch.tensor(projlist.reshape((trainlen, 256, 512, 1)), dtype=torch.float32).to(device)

	net_simple = simple().to(device)
	optimizer = optim.Adam(net_simple.parameters(), lr = 1)
	

	MODEL_FILENAME = '../../model/d2z_model'+"trained_model_simple.pth"
	if mode == 'train':
		train_model()
	else:
		net_simple.load_state_dict(torch.load(MODEL_FILENAME))
	
	test_model()
	pred_data = 'depth0.npy'
	visualize_result(pred_data)