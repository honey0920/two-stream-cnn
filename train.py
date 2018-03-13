from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import model, input_data

max_steps = 10000
batch_size = 128
ucf_path = '/home/honey/Downloads/UCF-101'

def train():
	image_holder = tf.placeholder(tf.float32, [batch_size, 224 ,224, 3])
	label_holder = tf.placeholder(tf.int32, [batch_size])
	#train_data, train_label, test_data, test_label = input_data.get_dataset(ucf_path)

	logits = model.inference(image_holder)
	loss = model.loss(logits, label_holder)

	train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)