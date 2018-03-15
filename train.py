from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
from six.moves import xrange
import tensorflow as tf
import os
import model
import model_vgg16

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'F:\\UCF-101\\train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('vgg_model', 'F:\\UCF-101\\model\\vgg16.npy',
                           """Directory where to write event logs """
                           """and checkpoint.""")
def train(use_vgg = False):
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        images, labels =model.distorted_inputs()

        if use_vgg:
            logits = model_vgg16.inference(images,FLAGS.vgg_model)
            loss = model_vgg16.loss(logits, labels)
        else:
            logits= model.inference(images)
            loss = model.loss(logits, labels)

        train_op = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(loss, global_step=global_step)

        saver = tf.train.Saver(tf.all_variables())
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start queue runners
        tf.train.start_queue_runners(sess=sess)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / duration
                sec_per_batch = float(duration)
                format_str = ('step %d,loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                if use_vgg:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'vgg_model.ckpt')
                else:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
    train(False)

if __name__ == '__main__':
    tf.app.run()
