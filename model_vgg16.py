import os
import numpy as np
import tensorflow as tf

NUM_CLASSES = 101


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
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
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
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def max_pool( bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def conv_layer (bottom, name, data_dict):
    with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
        conv = tf.nn.conv2d(bottom, data_dict[name][0], [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, data_dict[name][1]))
        return lout

def fc_layer( name, bottom, in_size, out_size):
    with tf.variable_scope(name) as scope:
        weight = _variable_with_weight_decay(name='weight',
                                             shape=[in_size, out_size],
                                             stddev=1e-3,
                                             wd=None)
        biases = _variable_on_cpu('biases',
                                  [out_size],
                                  tf.truncated_normal_initializer(stddev=1e-3, mean=0.0, dtype=tf.float32))
        reshape = tf.reshape(bottom, [-1, in_size])
        fc = tf.nn.bias_add(tf.matmul(reshape, weight), biases)
        return fc

def inference(images, npy_file):
    vgg_mean = [103.939, 116.779, 123.68]

    if not tf.gfile.Exists(npy_file):
        raise ValueError('Failed to find vgg model: ' + npy_file)

    data_dict = np.load(npy_file, encoding='latin1').item()
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=images)
    bgr = tf.concat(axis=3, values=[
        blue - vgg_mean[0],
        green - vgg_mean[1],
        red - vgg_mean[2],
    ])
    conv1_1 = conv_layer(bgr, "conv1_1",data_dict)
    conv1_2 = conv_layer(conv1_1, "conv1_2",data_dict)
    pool1 = max_pool(conv1_2, 'pool1')

    conv2_1 = conv_layer(pool1, "conv2_1",data_dict)
    conv2_2 = conv_layer(conv2_1, "conv2_2",data_dict)
    pool2 = max_pool(conv2_2, 'pool2')

    conv3_1 = conv_layer(pool2, "conv3_1",data_dict)
    conv3_2 = conv_layer(conv3_1, "conv3_2",data_dict)
    conv3_3 = conv_layer(conv3_2, "conv3_3",data_dict)
    pool3 = max_pool(conv3_3, 'pool3')

    conv4_1 = conv_layer(pool3, "conv4_1",data_dict)
    conv4_2 = conv_layer(conv4_1, "conv4_2",data_dict)
    conv4_3 = conv_layer(conv4_2, "conv4_3",data_dict)
    pool4 = max_pool(conv4_3,'pool4')

    conv5_1 = conv_layer(pool4, "conv5_1", data_dict)
    conv5_2 = conv_layer(conv5_1, "conv5_2", data_dict)
    conv5_3 = conv_layer(conv5_2, "conv5_3",data_dict)
    pool5 = max_pool(conv5_3, 'pool5')

    # detach original VGG fc layers and
    # reconstruct your own fc layers serve for your own purpose
    flatten = tf.reshape(pool5, [-1, 7 * 7 * 512])

    fc6 = fc_layer('fc6', flatten , 7 * 7 * 512, 4096)
    relu6 = tf.nn.dropout(tf.nn.relu(fc6), 0.5)

    fc7 = fc_layer('fc7', relu6, 4096, 2048)
    relu7 = tf.nn.dropout(tf.nn.relu(fc7), 0.5)
    output = fc_layer('output', relu7, 2048, NUM_CLASSES)
    return output

def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
