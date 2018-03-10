import tensorflow as tf

NUM_CLASSES = 101
CROP_SIZE = 224
CHANNELS = 3

def variablr_with_weight_loss(shape, stddev, wl):
	var = tf.Variable(tf.truncated_normal(shape,stddev = stddev))
	if wl is not None:
		weight_loss = tf.multiply(tf.nn.lr_loss(var), wl, name = 'weight_loss')
		tf.add_to_collection('losses', weight_loss)
	return var

def con2d(x,W, strides):
	return tf.nn.conv2d(x,W, strides = strides, padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding ='SAME')

image_holder = tf.placeholder(tf.float32, [batch_size, 224,224,3])
label_holder = tf.placeholder(tf.int32,[batch_size])

weight1 = variable_with_weight_loss(shape=[7,7,3,96], stddev = 5e-2, wl = 0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1,2,2,1], padding = 'SAME')
bias1 = tf.Variable(tf.constant(0.0, shape = [96]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
norm1 = tf.nn.lrn(conv1,4, bias = 1.0, alpha = 0.001/9.0, beta = 0.75)
pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')