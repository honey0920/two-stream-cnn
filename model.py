import tensorflow as tf

max_steps = 3000
batch_size = 128
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = input_data.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = input_data.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
	name: name of the variable
	shape: list of ints
	initializer: initializer for Variable
  Returns:
	Variable Tensor
  """
	with tf.device('/cpu:0'):
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
	name: name of the variable
	shape: list of ints
	stddev: standard deviation of a truncated Gaussian
	wd: add L2Loss weight decay multiplied by this float. If None, weight
		decay is not added for this Variable.
  Returns:
	Variable Tensor
  """
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	var = _variable_on_cpu(
		name,
		shape,
		tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def inference(images):
	with tf.variable_scope('conv1') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[7, 7, 3, 96],
											 stddev=5e-2,
											 wd=None)
		conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name=scope.name)
	norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
					name='norm1')
	pool1 = tf.nn.max_pool(norm1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
						 padding='SAME', name='pool1')
	

	with tf.variable_scope('conv2') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[5, 5, 96, 256],
											 stddev=5e-2,
											 wd=None)
		conv = tf.nn.conv2d(pool1, kernel, [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)
	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
					name='norm1')
	pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
						 padding='SAME', name='pool1')

	with tf.variable_scope('conv3') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[3, 3, 256, 512],
											 stddev=5e-2,
											 wd=None)
		conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv3 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv4') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[3, 3, 512, 512],
											 stddev=5e-2,
											 wd=None)
		conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(pre_activation, name=scope.name)

	with tf.variable_scope('conv5') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[3, 3, 512, 512],
											 stddev=5e-2,
											 wd=None)
		conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(pre_activation, name=scope.name)
	pool5 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
						 padding='SAME', name='pool1')

	with tf.variable_scope('fc6') as scope:
		reshape = tf.reshape(pool5, [batch_size, -1])
		dim = reshape.get_shape()[1].value
		weights = _variable_with_weight_loss(shape = [dim,4096],stddev = 0.04, wl = 0.004)
		biases = tf.Variable(tf.constant(0.1,shape = [4096]))
		fc6 = tf.nn.relu(tf.matmul(reshape,weights) + biases, name=scope.name)

	with tf.variable_scope('fc7') as scope:
		weights = tf.variable_with_weight_loss(shape = [4096,2048], stddev = 0.04, wl = 0.004)
		biases= tf.Variable(tf.constant(0.1, shape=[2048]))
		fc7 = tf.nn.relu(tf.matmul(fc6,weights)+ biases, name=scope.name)

	 with tf.variable_scope('softmax_linear') as scope:
		weights = variable_with_weight_loss(shape = [2048,101], stddev = 1/2048.0, wl = 0.0)
		biases = tf.Variable(tf.constant(0.0, shape = [101]), name=scope.name)
		softmax_linear = tf.add(tf.matmul(fc7, weights), biases)
	return softmax_linear

def loss(logits, labels):
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels, logits=logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	# The total loss is defined as the cross entropy loss plus all of the weight
	# decay terms (L2 loss).
	return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss, global_step):

  # Variables that affect learning rate.
	num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
								  global_step,
								  decay_steps,
								  LEARNING_RATE_DECAY_FACTOR,
								  staircase=True)


	loss_averages_op = _add_loss_summaries(total_loss)
	
	with tf.control_dependencies([loss_averages_op]):`
		opt = tf.train.GradientDescentOptimizer(lr)
		grads = opt.compute_gradients(total_loss)

  # Apply gradients.
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)


  # Track the moving averages of all trainable variables.
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

	# ?? trian_op = tf.train.AdamOptimizer(lr).minimize(loss)
	return train_op	



 

image_holder = tf.placeholder(tf.float32, [batch_size, 224,224,3])
label_holder = tf.placeholder(tf.int32,[batch_size])

