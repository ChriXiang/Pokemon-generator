from imageio import imread, imsave
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import cv2
import math
from copy import deepcopy

def resize_img():
	src = "/Users/xiangtiange1/desktop/pokemon" 
	dst = "./128_poke"
	#print(os.listdir(src))
	for each in os.listdir(src):
		if each == '.DS_Store' or each == '.floyddata':
			continue
		img = cv2.imread(os.path.join(src,each))
		img = cv2.resize(img,(128,128))
		cv2.imwrite(os.path.join(dst,each), img)

def random_crop():
	src = "./128_poke" 
	dst = "./crop_pad_image"
	#print(os.listdir(src))
	for each in os.listdir(src):

		img = cv2.imread(os.path.join(src,each))
		#crop top
		rep = deepcopy(img)
		for i in range(rep.shape[1]):
			suf = ''
			ans = np.zeros(img.shape)
			if len(np.unique(rep[0,:])) == 1 :
				rep = rep[1:,:]
				ans[:rep.shape[0],:] = rep
				rep = ans
				suf = "_top%d.png" % i
				name = each[:-4]+suf
				cv2.imwrite(os.path.join(dst,name), rep)
			else:
				break
		#crop bottom
		rep = deepcopy(img)
		for i in range(rep.shape[1]):
			suf = ''
			ans = np.zeros(img.shape)
			if len(np.unique(rep[-1,:])) == 1 :
				rep = rep[:-1,:]
				ans[img.shape[0]-rep.shape[0]:,:] = rep
				rep = ans
				suf = "_bottom%d.png" % i
				name = each[:-4]+suf
				cv2.imwrite(os.path.join(dst,name), rep)
			else:
				break
		#crop left
		rep = deepcopy(img)
		for i in range(rep.shape[1]):
			suf = ''
			ans = np.zeros(img.shape)
			if len(np.unique(rep[:,0])) == 1 :
				rep = rep[:,1:]
				ans[:,:rep.shape[1]] = rep
				rep = ans
				suf = "_left%d.png" % i
				name = each[:-4]+suf
				cv2.imwrite(os.path.join(dst,name), rep)
			else:
				break
		#crop right
		rep = deepcopy(img)
		for i in range(rep.shape[1]):
			suf = ''
			ans = np.zeros(img.shape)
			if len(np.unique(rep[:,-1])) == 1 :
				rep = rep[:,:-1]
				ans[:,img.shape[1]-rep.shape[1]:] = rep
				rep = ans
				suf = "_right%d.png" % i
				name = each[:-4]+suf
				cv2.imwrite(os.path.join(dst,name), rep)
			else:
				break

def flip_image():
	src1 = "./128_poke" 
	src2 = "./crop_pad_image"
	dst = "./final_poke"

	for each in os.listdir(src1):
		if each == '.DS_Store' or each == '.floyddata':
			continue
		img = cv2.imread(os.path.join(src1,each))
		h_img = cv2.flip(img, 1)
		name = each[:-4]+'_h.png'
		cv2.imwrite(os.path.join(dst,name), h_img)
		v_img = cv2.flip(img, 0)
		name = each[:-4]+'_v.png'
		cv2.imwrite(os.path.join(dst,name), v_img)

	for each in os.listdir(src2):
		if each == '.DS_Store' or each == '.floyddata':
			continue
		img = cv2.imread(os.path.join(src2,each))
		h_img = cv2.flip(img, 1)
		name = each[:-4]+'_h.png'
		cv2.imwrite(os.path.join(dst,name), h_img)
		v_img = cv2.flip(img, 0)
		name = each[:-4]+'_v.png'
		cv2.imwrite(os.path.join(dst,name), v_img)

def move_image():
	src1 = "./128_poke" 
	src2 = "./crop_pad_image"
	src3 = "./final_poke"
	dst = "./poke"
	for each in os.listdir(src3):
		if each == '.DS_Store' or each == '.floyddata':
			continue
		os.rename(os.path.join(src3,each),os.path.join(dst,each))

#uncomment and execute in sequnce

#resize_img()
#random_crop()
#flip_image()
move_image()