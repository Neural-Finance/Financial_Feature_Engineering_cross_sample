'''
This is the resnet structure
'''
import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages

fc_initializer = tf.contrib.layers.xavier_initializer
conv2d_initializer = tf.contrib.layers.xavier_initializer_conv2d


# create weight variable
def create_var(name, shape, initializer, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=tf.float32,
                           initializer=initializer, trainable=trainable)


# conv2d layer
def conv2d(x, num_outputs, kernel_size, stride=1, scope="conv2d"):
    num_inputs = x.get_shape()[-1]
    with tf.variable_scope(scope):
        kernel = create_var("kernel", [3, 3,
                                       num_inputs, num_outputs],
                            conv2d_initializer())
        return tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1],
                            padding="VALID")


# fully connected layer
def fc(x, num_outputs, scope="fc"):
    num_inputs = x.get_shape()[-1]
    with tf.variable_scope(scope):
        weight = create_var("weight", [num_inputs, num_outputs],
                            fc_initializer())
        bias = create_var("bias", [num_outputs, ],
                          tf.zeros_initializer())
        return tf.nn.xw_plus_b(x, weight, bias)


# batch norm layer
def batch_norm(x, decay=0.999, epsilon=1e-03, is_training=True,
               scope="scope"):
    x_shape = x.get_shape()
    num_inputs = x_shape[-1]
    reduce_dims = list(range(len(x_shape) - 1))
    with tf.variable_scope(scope):
        beta = create_var("beta", [num_inputs, ],
                          initializer=tf.zeros_initializer())
        gamma = create_var("gamma", [num_inputs, ],
                           initializer=tf.ones_initializer())
        # for inference
        moving_mean = create_var("moving_mean", [num_inputs, ],
                                 initializer=tf.zeros_initializer(),
                                 trainable=False)
        moving_variance = create_var("moving_variance", [num_inputs],
                                     initializer=tf.ones_initializer(),
                                     trainable=False)
    if is_training:
        mean, variance = tf.nn.moments(x, axes=reduce_dims)
        update_move_mean = moving_averages.assign_moving_average(moving_mean,
                                                                 mean, decay=decay)
        update_move_variance = moving_averages.assign_moving_average(moving_variance,
                                                                     variance, decay=decay)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_variance)
    else:
        mean, variance = moving_mean, moving_variance
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)


# avg pool layer
def avg_pool(x, pool_size, scope):
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(x, [1, pool_size, pool_size, 1],
                              strides=[1, pool_size, pool_size, 1], padding="SAME")


# max pool layer
def max_pool(x, pool_size, stride, scope):
    with tf.variable_scope(scope):
        return tf.nn.max_pool(x, [1, pool_size, pool_size, 1],
                              [1, stride, stride, 1], padding="SAME")


class LENET(object):
    def __init__(self):
        pass

    def inference(self, inputs, num_classes=2000, is_training=True,scope="lenet"):

        self.inputs = inputs
        self.is_training = is_training
        self.num_classes = num_classes
        graph = tf.get_default_graph()
        self.keep_rate = graph.get_tensor_by_name('Placeholder:0')

        #graph
        net = tf.nn.relu(conv2d(inputs, 16, 3, 2, scope="conv1"))
        net = max_pool(net, 3, 2, scope="maxpool1")

        net = tf.nn.relu(conv2d(net, 32, 3, 2, scope="conv2"))
        net = max_pool(net, 16, 2, scope="maxpool2")

        net = tf.nn.relu(conv2d(net, 32, 3, 2, scope="conv3"))
        net2 = max_pool(net, 16, 2, scope="maxpool3")

        net3 = tf.reshape(net2,[-1,net2.get_shape().as_list()[1]*net2.get_shape().as_list()[2]*net2.get_shape().as_list()[3]])
        l1 = tf.layers.dense(net3, num_classes, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
        self.out = tf.layers.dense(l1, num_classes, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))

        return self.out
