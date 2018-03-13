from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tensorflow as tf

file_path = 'F:\\UCF-101-image\\UCF-101'
IMAGE_SIZE = 224
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5000
NUM_EXAMPLE_PER_EPOCH_FOR_EVAL = 1000

def get_frame_path(videoname):
	image_path = ''
	for root,dirs,filenames in os.walk(videoname):
		index = random.randint(0,len(filenames) -1 )
		image_path = os.path.join(videoname, filenames[index])
		#img_raw =tf.gfile.FastGFile(image_name,'rb').read()
		#img = tf.image.decode_jpeg(img_raw).eval()
	return image_path

def read_data(input_queue):
	class dataRecord(object):
		pass
	result = dataRecord()

	result.label = input_queue[1]
	file_contents = tf.read_file(input_queue[0])
	result.image = tf.image.decode_jpeg(file_contents,channels=3)
	return result

def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
	num_preprpocess_threads = 4
	if shuffle:
		images, label_batch = tf.train.shuffle_batch(
			[image, label],
			batch_size = batch_size,
			num_threads=num_preprpocess_threads,
			capacity=min_queue_examples + 3 * batch_size,
			min_after_dequeue=min_queue_examples)
	else:
		images, label_batch = tf.train.batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprpocess_threads,
			capacity=min_queue_examples +3 * batch_size)
	return images, tf.reshape(label_batch, [batch_size])

def distorted_input(ucf_path, batch_size, category=None):
	image_list = []
	label_list = []
	actions = os.listdir(ucf_path)
	#print(actions) 
	for i in range(len(actions)):
		videos = os.listdir(os.path.join(ucf_path,actions[i]))
		#print(videos)
		for video in videos:
			image_path = get_frame_path(os.path.join(ucf_path , actions[i] , video))
			image_list.append(image_path)
			label_list.append(i)

	labels = tf.convert_to_tensor(label_list, dtype = tf.int32)
	images = tf.convert_to_tensor(image_list, dtype = tf.string)

	input_queue = tf.train.slice_input_producer([images, labels], shuffle = False)

	with tf.name_scope('data_augmentation'):
		height = IMAGE_SIZE
		width = IMAGE_SIZE

		read_input = read_data(input_queue)
		reshaped_image = tf.cast(read_input.image, tf.float32)

		distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
		distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
		distorted_image = tf.image.random_flip_left_right(distorted_image,1)


		min_fraction_of_examples_in_queue = 0.4
		min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *min_fraction_of_examples_in_queue)

		print('Filling queue with %d images before starting to train.' % min_queue_examples)
		return _generate_image_and_label_batch(distorted_image, read_input.label, min_queue_examples, batch_size, shuffle = False)


if __name__ == '__main__':


	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord = coord)
		result = distorted_input(file_path, 64)

		for i in range(10):
			example = sess.run(result)
			print('hello')

		coord.request_stop()
		coord.join(threads)


