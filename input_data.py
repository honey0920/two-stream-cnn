from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tensorflow as tf

def get_single_frame(videoname):
	for root,dirs,filenames in os.walk(videoname):
		index = random.randint(0,len(filenames) -1 )
		image_name = str(videoname) + '/' + str(filenames[index])
		img_raw =tf.gfile.FastGFile(image_name,'rb').read()
		#img = tf.image.decode_jpeg(img_raw).eval()
	return img_raw

def get_dataset(ucf_path):
	image_list = []
	label_list = []
	train_data = []
	train_label = []
	test_data = []
	test_label = []
	actions = os.listdir(ucf_path)
	#print(actions) 
	for i in range(len(actions)):
		videos = os.listdir(ucf_path + '/' + actions[i])
		#print(videos)
		for video in videos:
			img = get_single_frame(ucf_path + '/' + actions[i] + '/' + video)
			image_list.append(img)
			label_list.append(i)
	total_size = len(label_list)
	index  = range(total_size)
	random.shuffle(index)
	for i in range(total_size):
		if i % 4 == 0:
			test_data.append(image_list[index[i]])
			test_label.append(label_list[index[i]])
		else :
			train_data.append(image_list[index[i]])
			train_label.append(label_list[index[i]])
	return train_data, train_label, test_data. test_label


if __name__ == '__main__':
	get_dataset('/home/honey/Downloads/UCF-101')
	#a = test(4)

