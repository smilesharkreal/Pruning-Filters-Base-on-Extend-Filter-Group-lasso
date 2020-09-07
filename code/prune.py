import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import json
import logging
from time import gmtime, strftime, localtime
import os
import loadcifardata
import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import getopt
from pyecharts.charts import *
import pyecharts.options as echarts_opts
import math
import scipy.io as sio
# tf.disable_v2_behavior()
# 定义输出台参数
flags = tf.app.flags
flags.DEFINE_string("model", "ResNet", "A type of model. Possible options are: Vgg, ResNet.")
# flags.DEFINE_string("data_path", None,"Where the training/test data is stored.")
flags.DEFINE_string("restore_path", None, "Model input directory.")
flags.DEFINE_string("config_file", None, "Parameter config file.")
# flags.DEFINE_bool("display_prune_rate", False, "Display how many weight had been pruned.")
flags.DEFINE_string("regularizer", None, "Regularizer type L1,L2.")
flags.DEFINE_boolean("crossgroup", False, "use the crossgroup lass")
flags.DEFINE_boolean("grouplasso", False, "use the grouplasso")
flags.DEFINE_string("optimizer", 'NAG', "Optimizer of sgd: gd and adam and Moment.")
# flags.DEFINE_boolean("freeze_mode", False, "How to freeze zero weights.")
flags.DEFINE_boolean("statistic", True, "Statistic how many wight under statistic.")
flags.DEFINE_boolean("datadisplay", True, "Statistic table data.")
flags.DEFINE_boolean("store_weight", False, "store the weight.")
flags.DEFINE_boolean("train", True, "the model is runing for train or display other data.")
flags.DEFINE_string("dpfilename", None, "data display filename.")
flags.DEFINE_string("LR", 'step', "the learning rates change,such as step or Ex")
flags.DEFINE_boolean("reverse", False, "make div para")
flags.DEFINE_boolean("retrain", False, "retain the model")
flags.DEFINE_boolean("partial_retrain", False, "reliase partial parameter train")
flags.DEFINE_boolean("flag", False, "flag the filter")
flags.DEFINE_boolean("lookrate", False, "look pruning rate")
flags.DEFINE_boolean("lock", False, "lock")
flags.DEFINE_float("cg_value", 0.0, "determine cg value")
flags.DEFINE_float("gl_1", 0.0, "determine gl value")
flags.DEFINE_float("gl_2", 0.0, "determine gl value")
flags.DEFINE_float("epoch", 0.0, "determine training epoch")
FLAGS = flags.FLAGS


def zerout_gradients_for_zero_weights(grads_and_vars, weight_list, bias_list, filter_bool, bias_bool):
    """ zerout gradients for weights with zero values, so as to freeze zero weights
      Args:
          zero respective weight
    """
    # raise ValueError("the zerout fuction donsn't enforce")

    gradients, variables = zip(*grads_and_vars)
    zero_op = []
    zerout_gradients = []
    i = 0
    j = 0
    for gradient, variable in zip(gradients, variables):
        if gradient is None:
            zerout_gradients.append(None)
            continue
        if variable.name == weight_list[i].name:
            zero_op.append(tf.assign(variable, tf.where(filter_bool[i], variable, tf.zeros_like(variable))))
            # zero_op.append(tf.assign(gradient, tf.where(filter_bool[i], gradient, tf.zeros_like(gradient))))
            # variable = tf.where(filter_bool[i], variable, tf.zeros_like(variable))
            gradient = tf.where(filter_bool[i], gradient, tf.zeros_like(gradient))
        if variable.name == bias_list[j].name:
            zero_op.append(tf.assign(variable, tf.where(bias_bool[i], variable, tf.zeros_like(variable))))
            # zero_op.append(tf.assign(gradient, tf.where(bias_bool[i], gradient, tf.zeros_like(gradient))))
            # variable = tf.where(bias_bool[i], variable, tf.zeros_like(variable))
            gradient = tf.where(bias_bool[i], gradient, tf.zeros_like(gradient))
    return list(zip(gradients, variables)), zero_op


def L2_freeze(grads_and_vars, weight_list, bias_list, no_loss_weight, no_loss_bias):
    gradients, variables = zip(*grads_and_vars)
    zerout_gradients = []
    zero_op = []
    i = 0
    j = 0
    for gradient, variable in zip(gradients, variables):
        if gradient is None:
            zerout_gradients.append(None)
            continue
        t = 0
        if variable.name == weight_list[i].name:

            zero_op.append(tf.assign(variable, tf.where(no_loss_weight[i], variable, tf.zeros_like(variable))))
            # zero_op.append(tf.assign(gradient, tf.where(filter_bool[i], gradient, tf.zeros_like(gradient))))
            # variable = tf.where(filter_bool[i], variable, tf.zeros_like(variable))
            # gradient = tf.where(no_loss_weight[i], gradient, tf.zeros_like(gradient))
            zerout_gradients.append(tf.where(no_loss_weight[i], gradient, tf.zeros_like(gradient)))
            if i + 1 < len(weight_list):
                i += 1
            t = 1
        if variable.name == bias_list[j].name:

            zero_op.append(tf.assign(variable, tf.where(no_loss_bias[j], variable, tf.zeros_like(variable))))
            # zero_op.append(tf.assign(gradient, tf.where(bias_bool[i], gradient, tf.zeros_like(gradient))))
            # variable = tf.where(bias_bool[i], variable, tf.zeros_like(variable))
            # gradient = tf.where(no_loss_bias[j], gradient, tf.zeros_like(gradient))
            zerout_gradients.append(tf.where(no_loss_bias[j], gradient, tf.zeros_like(gradient)))
            if j + 1 < len(bias_list):
                j += 1
            t = 1
        if t == 0:
            zerout_gradients.append(gradient)

    return list(zip(zerout_gradients, variables)), zero_op


def heatdata_deal(weight):
    weight_list = []
    weight_actually = []
    heat_len = 0
    label = []
    min_ = 999
    max_ = -999
    for i in range(len(weight)):
        weight[i] = np.abs(weight[i].astype(np.float64))
        label_ = "conv" + str(i + 1)
        label.append(label_)
        temp = np.sum(weight[i], axis=(0, 1, 2))
        weight_actually.append(temp)
        # weight_list.append(np.sort(temp)) # get the filter
        weight_list.append(temp)

        if heat_len < weight[i].shape[-1]:
            heat_len = weight[i].shape[-1]
    weight_list_ = []
    for i in range(len(weight_list)):
        for j in range(heat_len):
            if j < weight_list[i].shape[-1]:
                weight_list_.append((j, i, weight_list[i][j]))
                if min_ > weight_list[i][j]:
                    min_ = weight_list[i][j]
                if max_ < weight_list[i][j]:
                    max_ = weight_list[i][j]
    return heat_len, label, weight_list_, int(min_), math.ceil(max_), weight_list, weight_actually


# resNet 模型
class ResNetModel(object):
    def __init__(self, x, num_residual_blocks, num_filter_base, class_num, is_training):
        # para = []
        layers = []
        shape = []

        # variable initial
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        # 卷积定义
        def conv2d(input, filter, strides, padding="SAME", name=None):
            return tf.nn.conv2d(input, filter, strides, padding=padding, name=name)  # padding="SAME"用零填充边界

        # 残差块
        def residual_block(x, output_channel, is_training, shape):
            '''
            定义残差块儿
            :param x: 输入tensor
            :param output_channel: 输出的通道数
            :return: tensor
            需要注意的是:每经过一个stage,通道数就要 * 2
            在同一个stage中,通道数是没有变化的
            '''
            input_channel = x.get_shape().as_list()[-1]  # 拿出 输入 tensor 的 最后一维:也就是通道数
            if input_channel * 2 == output_channel:
                increase_dim = True
                strides = [1, 2, 2, 1]  #
            elif input_channel == output_channel:
                increase_dim = False
                strides = [1, 1, 1, 1]
            else:
                raise Exception("input channel can't match output channel")

            kernel_1 = weight_variable(shape=[3, 3, input_channel, output_channel])
            bias_1 = weight_variable(shape=[output_channel])
            tf.add_to_collection("weight", kernel_1)
            tf.add_to_collection("bias", bias_1)
            # para.append(kernel_1)
            # para.append(bias_1)
            conv_1 = conv2d(x, kernel_1, strides=strides) + bias_1
            bn_1 = tf.layers.batch_normalization(conv_1, training=is_training)
            conv_1 = tf.nn.relu(bn_1)
            shape.append(conv_1.get_shape().as_list())
            shape.append(kernel_1.get_shape().as_list())

            kernel_2 = weight_variable(shape=[3, 3, conv_1.get_shape().as_list()[-1], output_channel])
            bias_2 = weight_variable(shape=[output_channel])
            tf.add_to_collection("weight", kernel_2)
            tf.add_to_collection("bias", bias_2)
            # para.append(kernel_2)
            # para.append(bias_2)
            conv_2 = conv2d(conv_1, kernel_2, strides=[1, 1, 1, 1]) + bias_2
            bn_2 = tf.layers.batch_normalization(conv_2, training=is_training)
            shape.append(conv_2.get_shape().as_list())
            shape.append(kernel_2.get_shape().as_list())
            # conv_2 = tf.nn.relu(bn_2)
            if increase_dim:  # 需要使用降采样
                # pooled_x 数据格式 [ None, image_width, image_height, channel ]
                # 要求格式 [ None, image_width, image_height, channel * 2 ]
                pooled_x = tf.layers.average_pooling2d(x, (2, 2), (2, 2), padding='valid')
                '''
                如果输出通道数是输入的两倍的话,需要增加通道数量.
                maxpooling 只能降采样,而不能增加通道数,
                所以需要单独增加通道数
                '''
                padded_x = tf.pad(pooled_x,  # 参数 2 ,在每一个通道上 加 pad
                                  [
                                      [0, 0],
                                      [0, 0],
                                      [0, 0],
                                      [input_channel // 2, input_channel // 2]  # 实际上就是 2倍input_channel,需要均分开
                                  ]
                                  )
            else:
                padded_x = x
            output_x = bn_2 + padded_x  # 就是 公式: H(x) = F(x) + x
            output_x = tf.nn.relu(output_x)
            return output_x

        with tf.name_scope("conv1"):
            kernel_1 = weight_variable([3, 3, 3, 16])
            bias_1 = weight_variable([16])
            tf.add_to_collection("weight", kernel_1)
            tf.add_to_collection("bias", bias_1)
            # para.append(kernel_1)
            # para.append(bias_1)
            conv_1 = conv2d(x, kernel_1, strides=[1, 1, 1, 1]) + bias_1
            bn = tf.layers.batch_normalization(conv_1, training=is_training)
            layer_1 = tf.nn.relu(bn)
            shape.append(conv_1.get_shape().as_list())
            shape.append(kernel_1.get_shape().as_list())
            # layer_1 = tf.nn.relu(conv2d(x,kernel_1,strides=[1,1,1,1])+bias_1,name='layer_1')
            layers.append(layer_1)
        for i in range(2, 5):
            for j in range(1, num_residual_blocks + 1):
                with tf.name_scope('conv%d_%d' % (i, j)):
                    conv = residual_block(layers[-1], num_filter_base * (2 ** (i - 2)), is_training, shape)
                    layers.append(conv)
        with tf.variable_scope('fc'):
            global_pool = tf.reduce_mean(layers[-1], [1, 2])  # 求平均值函数,参数二 指定 axis
            logits = tf.layers.dense(global_pool, class_num)
            layers.append(logits)
        self._out = layers[-1]
        self._shape = shape

    @property
    def output(self):
        return self._out

    @property
    def output_shape(self):
        return self._shape

class Lenet_Model(object):
    def __init__(self, x, keep_prob, is_training, config_params):
        # para = []
        shape = []
        self.config = config_params
        model_channel = config_params["Lenet_channel"]

        # 卷积定义
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

        def bias_variable(name, shape):
            initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
            return tf.Variable(name=name, initial_value=initial)

        def max_pool(input, k_size=1, stride=1, name=None):
            return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                                  padding='SAME', name=name)
        # 32*32
        with tf.name_scope("conv1"):
            W_conv1_1 = tf.get_variable('conv1', shape=[5, 5, 3, model_channel[0]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv1_1 = bias_variable("bias1_1", [model_channel[0]])
            tf.add_to_collection("weight", W_conv1_1)
            tf.add_to_collection("bias", b_conv1_1)
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(x, W_conv1_1) + b_conv1_1, training=is_training))
            shape.append(output.get_shape().as_list())
            shape.append(W_conv1_1.get_shape().as_list())
            output = max_pool(output, 2, 2, "pool1")
        # 14* 14
        # 16*16
        with tf.name_scope("conv2"):
            W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, model_channel[0], model_channel[1]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv1_2 = bias_variable("bias1_2", [model_channel[1]])
            output = tf.nn.relu(
                tf.layers.batch_normalization(conv2d(output, W_conv1_2) + b_conv1_2, training=is_training))
            tf.add_to_collection("weight", W_conv1_2)
            tf.add_to_collection("bias", b_conv1_2)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv1_2.get_shape().as_list())
            output = max_pool(output, 2, 2, "pool1")
            # out :16
        # 8*8
        with tf.name_scope("conv3"):
            W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, model_channel[1], model_channel[2]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv2_1 = bias_variable("bias2_1", [model_channel[2]])
            output = tf.nn.relu(
                tf.layers.batch_normalization(conv2d(output, W_conv2_1) + b_conv2_1, training=is_training))
            tf.add_to_collection("weight", W_conv2_1)
            tf.add_to_collection("bias", b_conv2_1)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv2_1.get_shape().as_list())
        # 4*4
        with tf.name_scope("conv4"):
            W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, model_channel[2], model_channel[3]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv2_2 = bias_variable("bias2_2", [model_channel[3]])
            output = tf.nn.relu(
                tf.layers.batch_normalization(conv2d(output, W_conv2_2) + b_conv2_2, training=is_training))
            tf.add_to_collection("weight", W_conv2_2)
            tf.add_to_collection("bias", b_conv2_2)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv2_2.get_shape().as_list())
            output = max_pool(output, 2, 2, "pool2")
            # out :8
        with tf.name_scope("conv5"):
            W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, model_channel[3], model_channel[4]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv3_1 = bias_variable("bias3_1", [model_channel[4]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv3_1) + b_conv3_1, training=is_training))
            tf.add_to_collection("weight", W_conv3_1)
            tf.add_to_collection("bias", b_conv3_1)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv3_1.get_shape().as_list())

        with tf.name_scope("conv6"):
            W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, model_channel[4], model_channel[5]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv3_2 = bias_variable("bias3_2", [model_channel[5]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv3_2) + b_conv3_2, training=is_training))
            tf.add_to_collection("weight", W_conv3_2)
            tf.add_to_collection("bias", b_conv3_2)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv3_2.get_shape().as_list())
        with tf.name_scope("conv7"):
            W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3,model_channel[5], model_channel[6]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv3_3 = bias_variable("bias3_3", [model_channel[6]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv3_3) + b_conv3_3, training=is_training))
            tf.add_to_collection("weight", W_conv3_3)
            tf.add_to_collection("bias", b_conv3_3)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv3_3.get_shape().as_list())
            output = max_pool(output, 2, 2, "pool3")
            # out :4
        with tf.name_scope("conv8"):
            W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3,model_channel[6], model_channel[7]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv4_1 = bias_variable("bias4_1", [model_channel[7]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv4_1) + b_conv4_1, training=is_training))
            tf.add_to_collection("weight", W_conv4_1)
            tf.add_to_collection("bias", b_conv4_1)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv4_1.get_shape().as_list())
        with tf.name_scope("conv9"):
            W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3,model_channel[7],model_channel[8]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv4_2 = bias_variable("bias4_2", [model_channel[8]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv4_2) + b_conv4_2, training=is_training))
            tf.add_to_collection("weight", W_conv4_2)
            tf.add_to_collection("bias", b_conv4_2)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv4_2.get_shape().as_list())
        with tf.name_scope("conv10"):
            W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, model_channel[8], model_channel[9]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv4_3 = bias_variable("bias4_3", [model_channel[9]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv4_3) + b_conv4_3, training=is_training))
            tf.add_to_collection("weight", W_conv4_3)
            tf.add_to_collection("bias", b_conv4_3)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv4_3.get_shape().as_list())
            output = max_pool(output, 2, 2)
            # out :2
        with tf.name_scope("conv11"):
            W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, model_channel[9], model_channel[10]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv5_1 = bias_variable("bias5_1", [model_channel[10]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv5_1) + b_conv5_1, training=is_training))
            tf.add_to_collection("weight", W_conv5_1)
            tf.add_to_collection("bias", b_conv5_1)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv5_1.get_shape().as_list())
        with tf.name_scope("conv12"):
            W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, model_channel[10], model_channel[11]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv5_2 = bias_variable("bias5_2", [model_channel[11]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv5_2) + b_conv5_2, training=is_training))
            tf.add_to_collection("weight", W_conv5_2)
            tf.add_to_collection("bias", b_conv5_2)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv5_2.get_shape().as_list())
        # 2*2
        with tf.name_scope("conv5"):
            W_conv5_3 = tf.get_variable('conv5', shape=[5, 5, model_channel[0], model_channel[4]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv5_3 = bias_variable("bias5", [model_channel[4]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv5_3) + b_conv5_3, training=is_training))
            tf.add_to_collection("weight", W_conv5_3)
            tf.add_to_collection("bias", b_conv5_3)
            # 池化
            output = max_pool(output, 2, 2)
            # 10 *10
            shape.append(output.get_shape().as_list())
            shape.append(W_conv5_3.get_shape().as_list())
            temp_shape = output.get_shape().as_list()
            output = tf.reshape(output,[-1,temp_shape[-1]*temp_shape[-2]*temp_shape[-3]])
            # output = tf.contrib.layers.flatten(output)
            # flatten_output = tf.reshape(output, [-1, 1 * 1 * model_channel[4]])
        with tf.name_scope("fc1"):
            W_fc1 = tf.get_variable('fc1', shape=[output.get_shape().as_list()[-1], model_channel[5]], initializer=tf.keras.initializers.he_normal())
            b_fc1 = bias_variable("fc1_b", [model_channel[5]])
            output = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(output, W_fc1) + b_fc1, training=is_training))

            output = tf.nn.dropout(output, keep_prob)
        # with tf.name_scope("fc2"):
        #     W_fc2 = tf.get_variable('fc2', shape=[model_channel[13], model_channel[14]], initializer=tf.keras.initializers.he_normal())
        #     b_fc2 = bias_variable('fc2_b', [model_channel[14]])
        #     output = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(output, W_fc2) + b_fc2, training=is_training))
        #     output = tf.nn.dropout(output, keep_prob)
        with tf.name_scope("fc2"):
            W_fc3 = tf.get_variable('fc2', shape=[model_channel[5], self.config["class_num"]],
                                    initializer=tf.keras.initializers.he_normal())
            b_fc3 = bias_variable('fc2_b', [self.config["class_num"]])
            output = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(output, W_fc3) + b_fc3, training=is_training))
            self._out = output
            self._shape = shape


    @property
    def output(self):
        return self._out

    @property
    def output_shape(self):
        return self._shape

class VggModel(object):
    def __init__(self, x, keep_prob, is_training, config_params):
        # para = []
        shape = []
        self.config = config_params
        vgg_channel = config_params["vgg_channel"]

        # 卷积定义
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def bias_variable(name, shape):
            initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
            return tf.Variable(name=name, initial_value=initial)

        def max_pool(input, k_size=1, stride=1, name=None):
            return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                                  padding='SAME', name=name)

        with tf.name_scope("conv1"):
            W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, vgg_channel[0]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv1_1 = bias_variable("bias1_1", [vgg_channel[0]])
            tf.add_to_collection("weight", W_conv1_1)
            tf.add_to_collection("bias", b_conv1_1)
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(x, W_conv1_1) + b_conv1_1, training=is_training))
            shape.append(output.get_shape().as_list())
            shape.append(W_conv1_1.get_shape().as_list())

        with tf.name_scope("conv2"):
            W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, vgg_channel[0], vgg_channel[1]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv1_2 = bias_variable("bias1_2", [vgg_channel[1]])
            output = tf.nn.relu(
                tf.layers.batch_normalization(conv2d(output, W_conv1_2) + b_conv1_2, training=is_training))
            tf.add_to_collection("weight", W_conv1_2)
            tf.add_to_collection("bias", b_conv1_2)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv1_2.get_shape().as_list())
            output = max_pool(output, 2, 2, "pool1")
            # out :16
        with tf.name_scope("conv3"):
            W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, vgg_channel[1], vgg_channel[2]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv2_1 = bias_variable("bias2_1", [vgg_channel[2]])
            output = tf.nn.relu(
                tf.layers.batch_normalization(conv2d(output, W_conv2_1) + b_conv2_1, training=is_training))
            tf.add_to_collection("weight", W_conv2_1)
            tf.add_to_collection("bias", b_conv2_1)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv2_1.get_shape().as_list())
        with tf.name_scope("conv4"):
            W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, vgg_channel[2], vgg_channel[3]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv2_2 = bias_variable("bias2_2", [vgg_channel[3]])
            output = tf.nn.relu(
                tf.layers.batch_normalization(conv2d(output, W_conv2_2) + b_conv2_2, training=is_training))
            tf.add_to_collection("weight", W_conv2_2)
            tf.add_to_collection("bias", b_conv2_2)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv2_2.get_shape().as_list())
            output = max_pool(output, 2, 2, "pool2")
            # out :8
        with tf.name_scope("conv5"):
            W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, vgg_channel[3], vgg_channel[4]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv3_1 = bias_variable("bias3_1", [vgg_channel[4]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv3_1) + b_conv3_1, training=is_training))
            tf.add_to_collection("weight", W_conv3_1)
            tf.add_to_collection("bias", b_conv3_1)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv3_1.get_shape().as_list())

        with tf.name_scope("conv6"):
            W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, vgg_channel[4], vgg_channel[5]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv3_2 = bias_variable("bias3_2", [vgg_channel[5]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv3_2) + b_conv3_2, training=is_training))
            tf.add_to_collection("weight", W_conv3_2)
            tf.add_to_collection("bias", b_conv3_2)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv3_2.get_shape().as_list())
        with tf.name_scope("conv7"):
            W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3,vgg_channel[5], vgg_channel[6]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv3_3 = bias_variable("bias3_3", [vgg_channel[6]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv3_3) + b_conv3_3, training=is_training))
            tf.add_to_collection("weight", W_conv3_3)
            tf.add_to_collection("bias", b_conv3_3)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv3_3.get_shape().as_list())
            output = max_pool(output, 2, 2, "pool3")
            # out :4
        with tf.name_scope("conv8"):
            W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3,vgg_channel[6], vgg_channel[7]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv4_1 = bias_variable("bias4_1", [vgg_channel[7]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv4_1) + b_conv4_1, training=is_training))
            tf.add_to_collection("weight", W_conv4_1)
            tf.add_to_collection("bias", b_conv4_1)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv4_1.get_shape().as_list())
        with tf.name_scope("conv9"):
            W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3,vgg_channel[7],vgg_channel[8]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv4_2 = bias_variable("bias4_2", [vgg_channel[8]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv4_2) + b_conv4_2, training=is_training))
            tf.add_to_collection("weight", W_conv4_2)
            tf.add_to_collection("bias", b_conv4_2)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv4_2.get_shape().as_list())
        with tf.name_scope("conv10"):
            W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, vgg_channel[8], vgg_channel[9]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv4_3 = bias_variable("bias4_3", [vgg_channel[9]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv4_3) + b_conv4_3, training=is_training))
            tf.add_to_collection("weight", W_conv4_3)
            tf.add_to_collection("bias", b_conv4_3)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv4_3.get_shape().as_list())
            output = max_pool(output, 2, 2)
            # out :2
        with tf.name_scope("conv11"):
            W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, vgg_channel[9], vgg_channel[10]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv5_1 = bias_variable("bias5_1", [vgg_channel[10]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv5_1) + b_conv5_1, training=is_training))
            tf.add_to_collection("weight", W_conv5_1)
            tf.add_to_collection("bias", b_conv5_1)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv5_1.get_shape().as_list())
        with tf.name_scope("conv12"):
            W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, vgg_channel[10], vgg_channel[11]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv5_2 = bias_variable("bias5_2", [vgg_channel[11]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv5_2) + b_conv5_2, training=is_training))
            tf.add_to_collection("weight", W_conv5_2)
            tf.add_to_collection("bias", b_conv5_2)
            shape.append(output.get_shape().as_list())
            shape.append(W_conv5_2.get_shape().as_list())
        with tf.name_scope("conv13"):
            W_conv5_3 = tf.get_variable('conv5_3', shape=[3, 3, vgg_channel[11], vgg_channel[12]],
                                        initializer=tf.keras.initializers.he_normal())
            b_conv5_3 = bias_variable("bias5_3", [vgg_channel[12]])
            output = tf.nn.relu(tf.layers.batch_normalization(conv2d(output, W_conv5_3) + b_conv5_3, training=is_training))
            tf.add_to_collection("weight", W_conv5_3)
            tf.add_to_collection("bias", b_conv5_3)
            # 池化
            output = max_pool(output, 2, 2)

            shape.append(output.get_shape().as_list())
            shape.append(W_conv5_3.get_shape().as_list())

            # output = tf.contrib.layers.flatten(output)
            flatten_output = tf.reshape(output, [-1, 1 * 1 * vgg_channel[12]])
        with tf.name_scope("fc1"):
            W_fc1 = tf.get_variable('fc1', shape=[vgg_channel[12], vgg_channel[13]], initializer=tf.keras.initializers.he_normal())
            b_fc1 = bias_variable("fc1_b", [vgg_channel[13]])
            output = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(flatten_output, W_fc1) + b_fc1, training=is_training))

            output = tf.nn.dropout(output, keep_prob)
        with tf.name_scope("fc2"):
            W_fc2 = tf.get_variable('fc2', shape=[vgg_channel[13], vgg_channel[14]], initializer=tf.keras.initializers.he_normal())
            b_fc2 = bias_variable('fc2_b', [vgg_channel[14]])
            output = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(output, W_fc2) + b_fc2, training=is_training))
            output = tf.nn.dropout(output, keep_prob)

        with tf.name_scope("fc3"):
            W_fc3 = tf.get_variable('fc3', shape=[vgg_channel[14], self.config["class_num"]],
                                    initializer=tf.keras.initializers.he_normal())
            b_fc3 = bias_variable('fc3_b', [self.config["class_num"]])
            output = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(output, W_fc3) + b_fc3, training=is_training))
            self._out = output
            self._shape = shape

    @property
    def output(self):
        return self._out

    @property
    def output_shape(self):
        return self._shape


class ModelDeploy(object):
    def __init__(self, y, output, config_params, lr,l2_flag):
        weight_list = tf.get_collection("weight")
        self.config_params = config_params
        self.lr = lr
        bias_list = tf.get_collection("bias")
        self.config_params = config_params
        self._bias_bool = None
        self._filter_bool = None
        self._num_save_filter = None
        self._zero_op = tf.constant(0, dtype=tf.float32)
        self._filter = None
        self._L2_freeze_flag = 0
        self.l2_flag = l2_flag
        self.no_loss_weight = []
        self.no_loss_bias = []
        self.count_ = []
        for i in range(len(weight_list)):
            l_shape = weight_list[i].get_shape().as_list()
            self.count_.append(tf.zeros(l_shape[-1]))
        self.filter_tensor = []
        self.channel_tensor = []
        # self.kw =kw
        #
        # if len(kw) == 2:
        #     self.no_loss_weight = kw[0]
        #     self.no_loss_bias = kw[1]
        #
        # for i in range(len(weight_list)):
        #     w_shape = weight_list[i].get_shape().as_list()
        #     b_shape = bias_list[i].get_shape().as_list()
        #     self.no_loss_weight.append(np.ones(w_shape))
        #     self.no_loss_bias.append(np.ones(b_shape))

        def L2_lasso(weight):
            return tf.nn.l2_loss(weight)
        def group_lasso(weight1, weight_coff1, weight_coff2):
            t1 = tf.reduce_sum(tf.abs(weight1),axis = [0,1,2])
            t2 = tf.reduce_sum(tf.abs(weight1),axis = [0,1,3])
            t1 = t1*t1
            t2 = t2*t2
            t1 = tf.sqrt(tf.reduce_sum(t1))
            t2 = tf.sqrt(tf.reduce_sum(t2))
            return t1 *weight_coff1 + t2*weight_coff2

            # weight = tf.square(weight1)
            # t1 = tf.reduce_sum(weight, axis=[0, 1, 2]) + tf.constant(1.0e-8)
            # t1_1 = t1[:-1]
            # t1_2 = t1[1:]
            # t2 = tf.reduce_sum(weight, axis=[0, 1, 3]) + tf.constant(1.0e-8)
            # t2_1 = t2[:-1]
            # t2_2 = t2[1:]
            #
            # t1 = tf.sqrt(t1_1 +t1_2)
            # t2 = tf.sqrt(t2_1 + t2_2)
            #
            # t1 = tf.reduce_sum(t1)
            # t2 = tf.reduce_sum(t2)
            # return t1 * weight_coff1 + t2 * weight_coff2

        def cross_group(weight1, weight2, weight_coff):
            t1 = tf.reduce_sum(tf.abs(weight1),axis=[0,1,2])
            t2 = tf.reduce_sum(tf.abs(weight2),axis = [0,1,3])
            t = tf.sqrt(tf.reduce_sum((t1+t2)*(t1+t2)))
            # t = tf.sqrt(tf.reduce_sum(t1*t1)+tf.reduce_sum(t2*t2))
            return weight_coff*t

            # t1 = tf.square(weight1)
            # t2 = tf.square(weight2)
            # t1 = tf.reduce_sum(t1, axis=[0, 1, 2]) + tf.constant(1.0e-8)
            # t2 = tf.reduce_sum(t2, axis=[0, 1, 3])
            #
            # # # 学习每层crossgroup值
            # #             # m1, var1 = tf.nn.moments(weight1, [0, 1, 2])
            # #             # m2, var2 = tf.nn.moments(weight2, [0, 1, 3])
            # #             # var3 = tf.expand_dims(var1 + var2, 0)
            # #             # out = tf.layers.dense(var3, var3.get_shape().as_list()[-1])
            # #             # out = tf.abs(tf.clip_by_norm(out, weight_coff))
            # #             # out = tf.squeeze(out)
            # #             # t = tf.sqrt((t1 + t2) * out)
            # t = tf.sqrt(t1 + t2)
            # t = tf.reduce_sum(t)  ##crossgroup的值
            # return t * weight_coff


        def resNet_group(weight1,weight2,weight3,weight4,coff):
            t1 = tf.square(weight1)
            t2 = tf.square(weight2)
            t3 = tf.square(weight3)
            shape_2 = weight2.get_shape().as_list()
            shape_4 = weight4.get_shape().as_list()
            if 2* shape_2[-1] == shape_4[-2]:
                weight4 =tf.nn.avg_pool(weight4,ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
            t4 = tf.square(weight4)
            t1 = tf.reduce_sum(t1, axis=[0, 1, 2])
            t2_1 = tf.reduce_sum(t2, axis=[0, 1, 3])
            t2_2 = tf.reduce_sum(t2, axis=[0, 1, 2])
            t3 = tf.reduce_sum(t3, axis=[0, 1, 3])
            t4 = tf.reduce_sum(t4, axis=[0, 1, 3])
            t = tf.sqrt(t1 + t2_1)+tf.sqrt(t2_2 + t3)+tf.sqrt(t2_2 + t4)
            t = tf.reduce_sum(t)
            return t*coff

        if FLAGS.regularizer == "L1":
            if config_params["L1para"] == None:
                raise ValueError("Must set --L1 parameter")
            else:
                # L1_list = [tf.contrib.layers.l1_regularizer(config_params["L1para"])(var) for var in
                #            tf.trainable_variables()]
                L1_list = None
                self._regularization = tf.add_n(L1_list)
        elif FLAGS.regularizer == "L2":
            if config_params["L2para"] == None:
                raise ValueError("Must set --L2 parameter")
            else:
                self._regularization = tf.constant(0, dtype=tf.float32)
                # L2_list = [tf.contrib.layers.l2_regularizer(config_params["L2para"])(var) for var in tf.trainable_variables()]
                for i in range(len(weight_list)):
                    self._regularization += L2_lasso(weight_list[i])
                self._regularization *= config_params["L2para"]

        else:
            self._regularization = tf.constant(0, dtype=tf.float32)
        #

        #  use crossgroup
        if FLAGS.crossgroup == True:
            cg_list = weight_list
            crossgroup_ = tf.constant(0, dtype=tf.float32)


            t = 1
            # for i in range(len(cg_list) - 1):
            #     # if i == 0:
            #     #     t =1.2
            #     # if i == 1 or i == 2 or i == 3:
            #     #     t = 4/3
            #     crossgroup_ += cross_group(cg_list[i], cg_list[i + 1], t * config_params["crossgroup_para"])

            if FLAGS.model == "ResNet" :
                for i in range(1,len(cg_list)-5,2):
                    crossgroup_ += resNet_group(cg_list[i],cg_list[i+1],cg_list[i+2],cg_list[i+4], t * config_params["crossgroup_para"])
                # for i in range(len(cg_list) - 4,len(cg_list)-1):
                #     crossgroup_ += cross_group(cg_list[i], cg_list[i + 1], t * config_params["crossgroup_para"])
            else:
                for i in range(len(cg_list) - 1):
                    # if i == 0:
                    #     t =1.2
                    # if i == 1 or i == 2 or i == 3:
                    #     t = 4/3
                    crossgroup_ += cross_group(cg_list[i], cg_list[i + 1], t * config_params["crossgroup_para"])
            self._crossgroup = crossgroup_
        else:
            self._crossgroup = tf.constant(0, dtype=tf.float32)

        # use grouplasso
        if FLAGS.grouplasso == True:
            gl_list = weight_list
            grouplasso_ = tf.constant(0, dtype=tf.float32)
            for i in range(len(gl_list)):
                grouplasso_ += group_lasso(gl_list[i], config_params["grouplasso_para1"],
                                           config_params["grouplasso_para2"])
            self._grouplasso = grouplasso_
        else:
            self._grouplasso = tf.constant(0, dtype=tf.float32)

        self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))

        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self._accuracy = accuracy_

        # 取得filter 和 channel
        def get_f_c():
            f = []
            c = []
            for i in range(len(weight_list)):
                f.append(tf.reduce_sum(tf.abs(weight_list[i]),axis=[0,1,2]))
                c.append(tf.reduce_sum(tf.abs(weight_list[i]),axis=[0,1,3]))
            return f,c
        self.filter_tensor ,self.channel_tensor = get_f_c()
        def get_extend_f():
            extend = []
            for i in range(len(weight_list)-1):
                extend.append(tf.reduce_sum(tf.abs(weight_list[i]),axis=[0,1,2])+tf.reduce_sum(tf.abs(weight_list[i+1]),axis=[0,1,3]))



    def filter_flag(self):
        weight_list = tf.get_collection("weight")  ##取得张量
        for i in range(len(weight_list) - 1):
            f1 = tf.reduce_sum(tf.abs(weight_list[i]), axis=[0, 1, 2], keepdims=True)
            f1 = tf.squeeze(f1)
            f2 = tf.reduce_sum(tf.abs(weight_list[i + 1]), axis=[0, 1, 3], keepdims=True)
            f2 = tf.squeeze(f2)
            f = f1 + f2
            f_bool = tf.greater(f, self.config_params["threshold"])
            f_value = tf.where(f_bool, tf.ones_like(f), tf.zeros_like(f))
            self.count_[i] += f_value
        f = tf.reduce_sum(tf.abs(weight_list[-1]), axis=[0, 1, 2], keepdims=True)
        ## real value
        f = tf.squeeze(f)
        f_bool = tf.greater(f, self.config_params["threshold"])
        f_value = tf.where(f_bool, tf.ones_like(f), tf.zeros_like(f))
        self.count_[i] += f_value


        # # single layer
        # for i in range(len(weight_list)):
        #     f = tf.reduce_sum(tf.abs(weight_list[i]), axis=[0, 1, 2], keepdims=True)
        #     ## real value
        #     f = tf.squeeze(f)
        #     f_bool = tf.greater(f,self.config_params["threshold"])
        #     f_value = tf.where(f_bool,tf.ones_like(f),tf.zeros_like(f))
        #     self.count_[i] += f_value

    def default_count(self):
        self.count_ = []
        weight_list = tf.get_collection("weight")
        for i in range(len(weight_list)):
            l_shape = weight_list[i].get_shape().as_list()
            self.count_.append(tf.zeros(l_shape[-1]))

    def freeze_op(self):
        # 统计过滤器低于阈值的个数
        weight_list = tf.get_collection("weight")
        bias_list = tf.get_collection("bias")
        assgin_op = []
        filter_bool_ = []
        bias_bool_ = []
        num_save_filter_ = []
        filter = []
        for i in range(len(weight_list)):
            l_shape = weight_list[i].get_shape().as_list()
            f = tf.reduce_sum(tf.abs(weight_list[i]), axis=[0, 1, 2], keepdims=True)
            ## real value
            filter.append(tf.squeeze(f))
            f_bool = tf.greater(f, self.config_params["threshold"])  # 大于阈值的布尔值
            num_filter = tf.count_nonzero(f_bool)  # 统计当层保留filter个数
            num_save_filter_.append(num_filter)  # 当层保留的filter数目
            b_bool = tf.squeeze(f_bool)
            f_bool = tf.tile(f_bool, [l_shape[0], l_shape[1], l_shape[2], 1])
            assgin_op.append(tf.assign(weight_list[i], tf.where(f_bool, weight_list[i], tf.zeros_like(weight_list[i]))))
            assgin_op.append(tf.assign(bias_list[i], tf.where(b_bool, bias_list[i], tf.zeros_like(bias_list[i]))))
            bias_bool_.append(b_bool)

            filter_bool_.append(f_bool)  # 当层大于阈值布尔值
        self._bias_bool = bias_bool_
        self._filter_bool = filter_bool_
        self._num_save_filter = num_save_filter_
        self._filter = filter
        # self._num_save_filter = num_save_filter_
        return assgin_op

    def get_weitht(self):
        return tf.get_collection("weight")

    def get_gl_state(self):
        return

    def get_bias(self):
        return tf.get_collection("bias")

    def filter(self):
        weight_actually = []
        weight = tf.get_collection("weight")
        for i in range(len(weight)):
            weight[i] = tf.abs(weight[i])
            temp = tf.reduce_sum(weight[i], axis=[0, 1, 2])
            weight_actually.append(temp)
        return weight_actually

    def pass_weight_bias(self, weight, bias):
        self.no_loss_weight = weight
        self.no_loss_bias = bias

    @property
    def accuracy(self):
        return self._accuracy

    # @property
    # def reverse(self):
    #     return self._reverse
    @property
    def cost(self):
        return self._cost

    @property
    def bias_bool(self):
        return self._bias_bool

    @property
    def num_save_filter(self):
        return self._num_save_filter

    @property
    def filter_bool(self):
        return self._filter_bool

    @property
    def regularization(self):
        return self._regularization

    @property
    def crossgroup(self):
        return self._crossgroup

    @property
    def grouplasso(self):
        return self._grouplasso

    ## filter值
    @property
    def filter_value(self):
        return self._filter

    @property
    def filter_count(self):
        return self.count_
    @property
    def train_op(self):
        if 'Moment' == FLAGS.optimizer:
            _optimizer = tf.train.MomentumOptimizer(self.lr, self.config_params["momentum"], use_nesterov=False)
        elif "NAG" == FLAGS.optimizer:
            _optimizer = tf.train.MomentumOptimizer(self.lr, self.config_params["momentum"], use_nesterov=True)
        elif 'gd' == FLAGS.optimizer:
            _optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif 'adam' == FLAGS.optimizer:
            _optimizer = tf.train.AdamOptimizer(self.lr)
        else:
            raise ValueError("Wrong optimizer!")
        if FLAGS.partial_retrain:
            ## 需要训练的参数
            loss_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.config_params["partial_scope"])
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = _optimizer.minimize(
                    self.cost + self.regularization + self.grouplasso + self.crossgroup, var_list=loss_vars)
            return self.train_step

        #  计算梯度
        _grads = _optimizer.compute_gradients(self.cost +self.l2_flag* self.regularization + self.grouplasso + self.crossgroup)

        # grads = optimizer.compute_gradients(self.cost + self.regularization + self.grouplasso + self.crossgroup)
        # def reverse(weight,para):
        #     weight = tf.square(weight)
        #     t = tf.reduce_sum(weight, axis=[0, 1, 2]) + tf.constant(1.0e-8)
        #     norm_t = tf.reduce_sum(t)

        # if FLAGS.reverse:
        #     reverse_ = tf.constant(0, dtype=tf.float32)
        #     for i in range(len(weight_list)):
        #         reverse_ = reverse(weight_list[i],config_params["reverse"])
        #     for i, (g, v) in enumerate(self._grads):
        #         if g is not None:
        #             self._grads[i] = (tf.clip_by_norm(g, config_params["norm_clip"]), v)
        # else:
        #     self._reverse = tf.constant(0, dtype=tf.float32)
        #
        # # 冻结梯度
        # if FLAGS.freeze_mode:
        #     filter_bool_ = []
        #     bias_bool_ = []
        #     num_save_filter_ = []
        #     for i in range(len(weight_list)):
        #         l_shape = weight_list[i].get_shape().as_list()
        #         f = tf.reduce_sum(tf.abs(weight_list[i]), axis=[0, 1, 2], keepdims=True)
        #         f_bool = tf.greater(f, self.config_params["threshold"])  # 大于阈值的布尔值
        #         num_filter = tf.count_nonzero(f_bool)  # 统计当层保留filter个数
        #         num_save_filter_.append(num_filter)  # 当层保留的filter数目
        #         b_bool = tf.squeeze(f_bool)
        #         f_bool = tf.tile(f_bool, [l_shape[0], l_shape[1], l_shape[2], 1])
        #         bias_bool_.append(b_bool)
        #         filter_bool_.append(f_bool)  # 当层大于阈值布尔值
        #     self._bias_bool = bias_bool_
        #     self._filter_bool = filter_bool_
        #     self._grads,zero_op = zerout_gradients_for_zero_weights(self._grads,weight_list,bias_list, self.filter_bool,self.bias_bool)
        #     self._zero_op = zero_op

        if FLAGS.retrain:
            _grads, zero_op = L2_freeze(_grads, tf.get_collection("weight"), tf.get_collection("bias"),
                                        self.no_loss_weight, self.no_loss_bias)
            self._zero_op = zero_op
        # 优化器更新梯度
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = _optimizer.apply_gradients(_grads)
        return self.train_step

    @property
    def zero_op(self):
        return self._zero_op

def run_training(sess, x, y_, train_x, train_y, cross_entropy, accuracy, is_training, cg, gl, reg, keep_prob):
    acc = 0.0
    loss = 0.0
    _cg = 0.0
    _gl = 0.0
    _reg = 0.0
    pre_index = 0
    add = 1000
    for it in range(50):
        batch_x = train_x[pre_index:pre_index + add]
        batch_y = train_y[pre_index:pre_index + add]
        pre_index = pre_index + add
        # loss_, acc_, cg_, gl_, reg_ = sess.run([cross_entropy, accuracy, cg, gl, reg],
        #                                        feed_dict={x: batch_x, y_: batch_y, is_training: False, keep_prob: 1.0})
        acc_ = sess.run(accuracy,feed_dict={x: batch_x, y_: batch_y, is_training: False, keep_prob: 1.0})

        # loss += loss_ / 50.0
        acc += acc_ / 50.0
        # _cg += cg_ / 50.0
        # _gl += gl_ / 50.0
        # _reg += reg_ / 50.0

    return acc, loss, _reg, _cg, _gl

def run_testing(sess, x, y_, test_x, test_y, cross_entropy, accuracy, is_training, cg, gl, reg, keep_prob):
    acc = 0.0
    loss = 0.0
    _cg = 0.0
    _gl = 0.0
    _reg = 0.0
    pre_index = 0
    add = 1000
    if FLAGS.model == "Lenet" or FLAGS.model == "LeNet":
        add = 1627
        for it in range(16):
            batch_x = test_x[pre_index:pre_index + add]
            batch_y = test_y[pre_index:pre_index + add]
            pre_index = pre_index + add
            loss_, acc_, cg_, gl_, reg_ = sess.run([cross_entropy, accuracy, cg, gl, reg],
                                                   feed_dict={x: batch_x, y_: batch_y, is_training: False,
                                                              keep_prob: 1.0})
            loss += loss_ / 16.0
            acc += acc_ / 16.0
            _cg += cg_ / 16.0
            _gl += gl_ / 16.0
            _reg += reg_ / 16.0
    else:
        for it in range(10):
            batch_x = test_x[pre_index:pre_index + add]
            batch_y = test_y[pre_index:pre_index + add]
            pre_index = pre_index + add
            loss_, acc_, cg_, gl_, reg_ = sess.run([cross_entropy, accuracy, cg, gl, reg],
                                                   feed_dict={x: batch_x, y_: batch_y, is_training: False,
                                                              keep_prob: 1.0})
            loss += loss_ / 10.0
            acc += acc_ / 10.0
            _cg += cg_ / 10.0
            _gl += gl_ / 10.0
            _reg += reg_ / 10.0

    return acc, loss, _reg, _cg, _gl


def findmax(acc):
    l = list(acc)
    m = max(l)
    index = l.index(m)
    return index, m

def statistic_info(weight):
    L1_value = np.sum(np.abs(weight))
    mean = np.mean(weight,axis=(0,1,2))
    mean = [x  for x in mean if x != 0]
    mean = np.mean(mean)
    return L1_value,mean


def use_flag_schedule(ep, config_params):
    use_flag = config_params["learning_rates"]["use_flag"]
    phase = config_params["learning_rates"]["phase"]
    for i in range(len(phase)):
        if ep <= phase[i]:
            return use_flag[i]


def learning_rate_schedule(ep, config):
    phase = config["learning_rates"]["phase"]
    learning_rate = config["learning_rates"]["value"]
    for i in range(len(phase)):
        if ep <= phase[i]:
            return learning_rate[i]


def larning_rate_retrain(ep, config):
    phase = config["retrain_learning_rates"]["phase"]
    learning_rate = config["retrain_learning_rates"]["value"]
    for i in range(len(phase)):
        if ep <= phase[i]:
            return learning_rate[i]

def learning_rate_partial_retrain(ep,config):
    phase = config["partial_retrain_learning_rates"]["phase"]
    learning_rate = config["partial_retrain_learning_rates"]["value"]
    for i in range(len(phase)):
        if ep <= phase[i]:
            return learning_rate[i]

def lr_exponential(ep, config, global_step):
    phase = config["learning_rates"]["phase"]
    learning_rate = config["learning_rates"]["value"]
    lr = tf.train.exponential_decay(learning_rate[0], global_step, config["decay_steps"], config["decay_rate"])
    return lr

def cycle_larning_rate(iteration , config_params):
    base_lr = config_params["base_lr"]
    max_lr = config_params["max_lr"]
    stepsize = config_params["stepsize"]
    cycle = np.floor(1 + iteration / (2 * stepsize))
    x = np.abs(iteration / stepsize - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))
    return lr

# class ResNetconfig(object):
#     def __init__(self,config):
#         self.image_size = config["image_size"]
#         self.image_channel = config["image_channel"]
#         self.num_residual_blocks = config["num_residual_blocks"]
#         self.num_filter_base = config["num_filter_base"]
#         self.phase = config["learning_rates"]["phase"]
#         self.learning_rate =config["learning_rates"]["value"]
#
# class Vggconfig(object):
#     def __init__(self,config):
#         self.image_size = 32
#         self.image_channel = 3
#         self.learning_rates = {"phase": [80, 140, 180], "value": [0.1, 0.01, 0.001]}
# def get_config(model,config):
#     if model=="ResNet":
#         return ResNetconfig(config)
#     elif model == "Vgg":
#         return Vggconfig(config)
#     else:
#         raise ValueError("the model hasn't been config")

def store_weight_method(config_params, sess, logger):
    weight = tf.get_collection("weight")
    bias = tf.get_collection("bias")
    h5path = os.path.join(config_params['save_path'], FLAGS.model + "para.h5")

    while os.path.exists(h5path):
        if h5path[-1].isdigit() and h5path[-2] != "h":
            h5path = h5path[:-1] + str(int(h5path[-1]) + 1)
        else:
            h5path = h5path + "_1"

    f = h5py.File(h5path, 'w')
    for i in range(len(weight)):
        str_label = "weight" + str(i)
        str_bias = "bias" + str(i)
        f[str_label] = sess.run(weight[i])
        f[str_bias] = sess.run(bias[i])
    f.close()
    logger.info("保存参数在%s" % (h5path))


def statistic_method(sess, x, y_, test_x, test_y, cross_entropy, accuracy,
                     is_training, cg, gl, reg, keep_prob, logger, md, model, config_params,
                     max_to_keep, saver, log_file, title):
    # 取得最大模型
    ckpt = tf.train.get_checkpoint_state(config_params['save_path'] + "/")
    max_path = 0 - max_to_keep
    saver.restore(sess, ckpt.all_model_checkpoint_paths[max_path])
    # max_path = 0 - max_to_keep
    # saver.restore(sess, ckpt.all_model_checkpoint_paths[max_path])
    # ckpt = tf.train.latest_checkpoint(config_params['save_path']+"/")
    # saver.restore(sess, ckpt)

    sava_val_acc, val_cost, val_reg, val_cg, val_gl = run_testing(sess, x, y_, test_x, test_y,
                                                                  cross_entropy,
                                                                  accuracy, is_training, cg, gl, reg,
                                                                  keep_prob)
    logger.info("模型的识别率为%.4f" % (sava_val_acc))

    # use pyecharts

    weight_list_ = sess.run(md.get_weitht())  # get the weight kernel list
    heat_len, label, weight_list, min_, max_, weight_data, weight_actually = heatdata_deal(weight_list_)

    heatmap = HeatMap()
    heatmap.add_xaxis(list(range(1, heat_len + 1, 1)))
    heatmap.add_yaxis(series_name="kernel", yaxis_data=label, value=weight_list,
                      )
    # markline_opts = echarts_opts.MarkLineOpts(data=[echarts_opts.MarkLineItem(name="过滤器个数", x=64, symbol="none"),
    #                                                 echarts_opts.MarkLineItem(name="过滤器个数", x=128, symbol="none"),
    #                                                 echarts_opts.MarkLineItem(name="过滤器个数", x=256, symbol="none"),
    #                                                 echarts_opts.MarkLineItem(name="过滤器个数", x=512, symbol="none")])
    heatmap.set_global_opts(title_opts=echarts_opts.TitleOpts(title="filter heatmap", pos_left="50%"),
                            visualmap_opts=echarts_opts.VisualMapOpts(min_=min_, max_=max_,
                                                                      range_color=["black", "brown", "red"],
                                                                      pos_right="13%", pos_bottom="15%"),
                            legend_opts=echarts_opts.LegendOpts(pos_left="20%", is_show=False),
                            toolbox_opts=echarts_opts.ToolboxOpts(),
                            xaxis_opts=echarts_opts.AxisOpts(name="numb of filter", type_="value",
                                                             axisline_opts=echarts_opts.AxisLineOpts(
                                                                 symbol=['none', 'arrow']),
                                                             split_number=4,
                                                             is_scale=True,
                                                             name_gap=1, ),
                            yaxis_opts=echarts_opts.AxisOpts(name="layers")
                            )
    box = (Boxplot()
           .add_xaxis(label)
           .add_yaxis("kernel", Boxplot().prepare_data(weight_data))
           .set_global_opts(title_opts=echarts_opts.TitleOpts(title="convolution kernel boxplot", pos_left="50%"),
                            legend_opts=echarts_opts.LegendOpts(is_show=False),
                            toolbox_opts=echarts_opts.ToolboxOpts(),
                            yaxis_opts=echarts_opts.AxisOpts(name="value", axisline_opts=echarts_opts.AxisLineOpts(
                                symbol=['none', 'arrow'])),
                            xaxis_opts=echarts_opts.AxisOpts(name="numb of filter",
                                                             axisline_opts=echarts_opts.AxisLineOpts(
                                                                 symbol=['none', 'arrow']),
                                                             name_gap=1)
                            )
           )

    # try to prune the filter and no error loss
    weight_list_tensor = md.get_weitht()
    bias_list_tensor = md.get_bias()
    ori_total = 0.0
    prune_total = 0.0
    shape = model.output_shape
    no_loss_acc = []
    no_loss_mac = []
    filter_save = []
    filter_ori = []
    s_L1 = []
    s_mean = []
    # no_loss_weight = []
    # no_loss_bias = []
    logger.info('---------------------------无损精度查看-----------------------')
    for i in range(len(weight_actually)):
        temp_weight = sess.run(weight_list_tensor[i])
        temp_bias = sess.run(bias_list_tensor[i])
        l_shape = weight_list_tensor[i].get_shape().as_list()
        filter_ori.append(len(weight_data[i]))
        layer_th = 1
        _set_i = 1
        loss_flag = 0
        while layer_th > 0.0000001:
            bias_bool = weight_actually[i] >= layer_th
            weight_bool = np.tile(bias_bool, [l_shape[0], l_shape[1], l_shape[2], 1])
            sess.run(tf.assign(weight_list_tensor[i],
                               tf.where(weight_bool, weight_list_tensor[i],
                                        tf.zeros_like(weight_list_tensor[i]))))
            sess.run(tf.assign(bias_list_tensor[i],
                               tf.where(bias_bool, bias_list_tensor[i],
                                        tf.zeros_like(bias_list_tensor[i]))))
            cur_acc, _, _, _, _ = run_testing(sess, x, y_, test_x, test_y, cross_entropy, accuracy,
                                              is_training,
                                              cg, gl, reg,
                                              keep_prob)

            if cur_acc >= sava_val_acc:
                filter_save.append(np.count_nonzero(bias_bool))  # statistic the sava filter numb
                no_loss_acc.append(cur_acc)
                loss_flag = 1
                break
            else:
                if _set_i % 2 == 1:
                    layer_th /= 2
                else:
                    layer_th /= 5
                _set_i += 1
                sess.run(tf.assign(weight_list_tensor[i], temp_weight))
                sess.run(tf.assign(bias_list_tensor[i], temp_bias))


        if loss_flag == 0:
            if len(no_loss_acc) == 0:
                no_loss_acc.append(sava_val_acc)
            else:
                no_loss_acc.append(no_loss_acc[-1])
            filter_save.append(temp_weight.shape[-1])
            sess.run(tf.assign(weight_list_tensor[i], temp_weight))
            sess.run(tf.assign(bias_list_tensor[i], temp_bias))

        # L1_value,mean = statistic_info(sess.run(weight_list_tensor[i]))
        # s_L1.append(L1_value)
        # s_mean.append(mean)

        # no_loss_weight.append(sess.run(weight_list_tensor[i]))
        # no_loss_bias.append(sess.run(bias_list_tensor[i]))

        logger.info('第%d个卷积层原有过滤%d个，阈值为:%f，剩余过滤器有%d，剪枝后的精度为：%.4f' % (
            i + 1, temp_weight.shape[3], layer_th, filter_save[-1], cur_acc))
        orig_flops = shape[2 * i][1] * shape[2 * i][2] * 9 * shape[2 * i + 1][2] * shape[2 * i + 1][3]
        logger.info('当层原有FLOPs%.4f:' % orig_flops)
        ori_total += orig_flops
        if i == 0:
            save_flops = shape[2 * i][1] * shape[2 * i][2] * 9 * filter_save[-1] * 3
            prune_total += save_flops
        else:
            save_flops = shape[2 * i][1] * shape[2 * i][2] * 9 * filter_save[-1] * filter_save[-2]
            prune_total += save_flops
        logger.info('当层剪枝剩余FLOPs:%.4f，剪枝率为%.4f' % (orig_flops, 1 - save_flops / orig_flops))
        no_loss_mac.append(1 - save_flops / orig_flops)
    print(s_L1,s_mean)
    no_loss_acc.append(cur_acc)
    if FLAGS.model == "ResNet":
        ori_total += shape[-1][-1] * config_params["class_num"]
        prune_total += filter_save[-1] * config_params["class_num"]
    elif FLAGS.model == "Vgg":
        ori_total += shape[-1][-1] * shape[-2][1] * shape[-2][2] * 4096 + 4096 * 4096 + 4096 * \
                     config_params["class_num"]
        prune_total += filter_save[-1] * shape[-2][1] * shape[-2][2] * 4096 + 4096 * 4096 + 4096 * \
                       config_params["class_num"]
    elif FLAGS.model == "Lenet" or FLAGS.model == "LeNet":
        ori_total += shape[-1][-1] * shape[-2][1] * shape[-2][2] * 256 + 256 * \
                     config_params["class_num"]
        prune_total += filter_save[-1] * shape[-2][1] * shape[-2][2] * 256 +  256 * \
                       config_params["class_num"]
    else:
        raise ValueError("the other model doesn't come true")
    logger.info('卷积网络FLOPs剪枝率为：%.4f' % (1 - prune_total / ori_total))
    no_loss_mac.append(1 - prune_total / ori_total)
    logger.info('---------------------------无损精度查看完毕-----------------------')

    no_loss_acc = [float("%.2f" % (i * 100)) for i in no_loss_acc]
    no_loss_mac = [float("%.2f" % (i * 100)) for i in no_loss_mac]
    bar = (Bar()
           .add_xaxis(label)
           .add_yaxis(series_name="orignal filter", y_axis=filter_ori)
           .add_yaxis(series_name="remain filter", y_axis=filter_save, gap="-100%",
                      label_opts=echarts_opts.LabelOpts(position="inside"))
           .set_global_opts(title_opts=echarts_opts.TitleOpts(title="pruning filter with no loss", pos_left="50%"),
                            toolbox_opts=echarts_opts.ToolboxOpts(),
                            yaxis_opts=echarts_opts.AxisOpts(name="numb of filter",
                                                             axisline_opts=echarts_opts.AxisLineOpts(
                                                                 symbol=['none', 'arrow'])
                                                             ),
                            legend_opts=echarts_opts.LegendOpts(pos_left="20%"),
                            xaxis_opts=echarts_opts.AxisOpts(name="layers",
                                                             axisline_opts=echarts_opts.AxisLineOpts(
                                                                 symbol=['none', 'arrow'])
                                                             )))
    linelabel = label[:]
    linelabel.append("Totally")
    line = (Line().add_xaxis(linelabel)
            .add_yaxis("pruning MACs", no_loss_mac, label_opts=echarts_opts.LabelOpts(is_show=False))
            .add_yaxis("acc", no_loss_acc,
                       label_opts=echarts_opts.LabelOpts(is_show=False),
                       markline_opts=echarts_opts.MarkLineOpts(
                           data=[echarts_opts.MarkLineItem(name="baseline_acc", y=sava_val_acc * 100)]))
            .set_global_opts(title_opts=echarts_opts.TitleOpts(title="Macs and acc with prune no loss", pos_left="40%"),
                             toolbox_opts=echarts_opts.ToolboxOpts(),
                             yaxis_opts=echarts_opts.AxisOpts(name="rate(%)",
                                                              axisline_opts=echarts_opts.AxisLineOpts(
                                                                  symbol=['none', 'arrow'])),
                             legend_opts=echarts_opts.LegendOpts(is_show=True, pos_left="10%"),
                             xaxis_opts=echarts_opts.AxisOpts(name="layers",
                                                              axistick_opts=echarts_opts.AxisTickOpts(
                                                                  is_align_with_label=True, is_show=True),
                                                              is_scale=True,
                                                              axislabel_opts=echarts_opts.LabelOpts(interval=0,
                                                                                                    rotate=-45),
                                                              axisline_opts=echarts_opts.AxisLineOpts(
                                                                  symbol=['none', 'arrow']),
                                                              axispointer_opts=echarts_opts.AxisPointerOpts(
                                                                  is_show=True)

                                                              ),
                             )
            )

    line_sta = (Line().add_xaxis(label)
            .add_yaxis("L1_value", s_L1, label_opts=echarts_opts.LabelOpts(is_show=False))
            .add_yaxis("conv_mean", s_mean,
                       label_opts=echarts_opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=echarts_opts.TitleOpts(title="Conv L1_value and mean", pos_left="40%"),
                             toolbox_opts=echarts_opts.ToolboxOpts(),
                             yaxis_opts=echarts_opts.AxisOpts(name="value"),
                                                              # axisline_opts=echarts_opts.AxisLineOpts(
                                                              #     symbol=['none', 'arrow'])),
                             legend_opts=echarts_opts.LegendOpts(is_show=True, pos_left="10%"),
                             xaxis_opts=echarts_opts.AxisOpts(name="layers",
                                                              axistick_opts=echarts_opts.AxisTickOpts(
                                                                  is_align_with_label=True, is_show=True),
                                                              is_scale=True,
                                                              axislabel_opts=echarts_opts.LabelOpts(interval=0,
                                                                                                    rotate=-45),
                                                              axisline_opts=echarts_opts.AxisLineOpts(
                                                                  symbol=['none', 'arrow']),
                                                              axispointer_opts=echarts_opts.AxisPointerOpts(
                                                                  is_show=True)

                                                              ),
                             )
            )
    # 重新加载
    ckpt = tf.train.get_checkpoint_state(config_params['save_path'] + "/")
    max_path = 0 - max_to_keep
    saver.restore(sess, ckpt.all_model_checkpoint_paths[max_path])
    op = md.freeze_op()
    num_filter = md.num_save_filter
    ori_total = 0.0
    prune_total = 0.0
    shape = model.output_shape
    filter_ori_ = []
    th_acc = []
    th_mac = []
    if FLAGS.model == "ResNet":
        ori_total += shape[-1][-1] * config_params["class_num"]
        prune_total += sess.run(num_filter[-1]) * config_params["class_num"]
    elif FLAGS.model == "Vgg":
        ori_total += shape[-1][-1] * shape[-2][1] * shape[-2][2] * 4096 + 4096 * 4096 + 4096 * \
                     config_params["class_num"]
        prune_total += sess.run(num_filter[-1]) * shape[-2][1] * shape[-2][
            2] * 4096 + 4096 * 4096 + 4096 * config_params["class_num"]
    elif FLAGS.model == "Lenet" or FLAGS.model == "LeNet":
        ori_total += shape[-1][-1] * shape[-2][1] * shape[-2][2] * 256 + 256 * \
                     config_params["class_num"]
        prune_total += filter_save[-1] * shape[-2][1] * shape[-2][2] * 256 +  256 * \
                       config_params["class_num"]
    else:
        raise ValueError("the other model doesn't come true")

    logger.info('------------------阈值剪枝情况-------------------')
    for i in range(0, len(op), 2):
        sess.run(op[i])  # 将低于阈值的权重设置为0
        sess.run(op[i + 1])
        val_acc, val_cost, val_reg, val_cg, val_gl = run_testing(sess, x, y_, test_x, test_y,
                                                                 cross_entropy,
                                                                 accuracy, is_training, cg, gl, reg,
                                                                 keep_prob)
        th_acc.append(val_acc)
        logger.info('----------------------------------------------------')
        logger.info('第%d个卷积层原有过滤%d个，阈值为:%f，剩余过滤器有%d，' % (
            i // 2 + 1, shape[i + 1][3], config_params["threshold"], sess.run(num_filter[i // 2])))
        logger.info('剪枝后的精度为：%.4f' % (val_acc))
        filter_ori_.append(int(sess.run(num_filter[i // 2])))
        # 计算当层FLOPs值
        orig_flops = shape[i][1] * shape[i][2] * 9 * shape[i + 1][2] * shape[i + 1][3]
        ori_total += orig_flops
        if i == 0:
            save_flops = shape[i][1] * shape[i][2] * 9 * sess.run(num_filter[i // 2]) * 3
            prune_total += save_flops
        else:
            save_flops = shape[i][1] * shape[i][2] * 9 * sess.run(num_filter[i // 2 - 1]) * sess.run(
                num_filter[i // 2])
            prune_total += save_flops
        logger.info('当层FLOPs剪枝率为：%.4f' % (1 - save_flops / orig_flops))
        th_mac.append(1 - save_flops / orig_flops)
    th_acc.append(val_acc)
    th_mac.append(1 - prune_total / ori_total)
    logger.info('卷积网络FLOPs剪枝率为：%.4f' % (1 - prune_total / ori_total))
    logger.info('----------------------------------------------------')
    th_acc = [float("%.2f" % (th * 100)) for th in th_acc]
    th_mac = [float("%.2f" % (th * 100)) for th in th_mac]
    bar1 = (Bar()
            .add_xaxis(label)
            .add_yaxis(series_name="orignal filter", y_axis=filter_ori)
            .add_yaxis(series_name="remain filter", y_axis=filter_ori_, gap="-100%",
                       label_opts=echarts_opts.LabelOpts(position="inside"))
            .set_global_opts(title_opts=echarts_opts.TitleOpts(title="pruning filter with threshold", pos_left="50%"),
                             toolbox_opts=echarts_opts.ToolboxOpts(),
                             yaxis_opts=echarts_opts.AxisOpts(name="numb of filter",
                                                              axisline_opts=echarts_opts.AxisLineOpts(
                                                                  symbol=['none', 'arrow'])
                                                              ),
                             legend_opts=echarts_opts.LegendOpts(pos_left="20%"),
                             xaxis_opts=echarts_opts.AxisOpts(name="layers",
                                                              axisline_opts=echarts_opts.AxisLineOpts(
                                                                  symbol=['none', 'arrow'])
                                                              ))
            )
    line1 = (Line().add_xaxis(linelabel)
             .add_yaxis("pruning MACs", th_mac, label_opts=echarts_opts.LabelOpts(is_show=False))
             .add_yaxis("acc", th_acc,
                        label_opts=echarts_opts.LabelOpts(is_show=False),
                        markline_opts=echarts_opts.MarkLineOpts(
                            data=[echarts_opts.MarkLineItem(name="baseline_acc", y=sava_val_acc * 100)]))
             .set_global_opts(
        title_opts=echarts_opts.TitleOpts(title="Macs and acc of prune threshold", pos_left="40%"),
        toolbox_opts=echarts_opts.ToolboxOpts(),
        yaxis_opts=echarts_opts.AxisOpts(name="rate(%)",
                                         axisline_opts=echarts_opts.AxisLineOpts(
                                             symbol=['none', 'arrow'])),
        legend_opts=echarts_opts.LegendOpts(is_show=True, pos_left="10%"),
        xaxis_opts=echarts_opts.AxisOpts(name="layers",
                                         axistick_opts=echarts_opts.AxisTickOpts(
                                             is_align_with_label=True, is_show=True),
                                         is_scale=True,
                                         axislabel_opts=echarts_opts.LabelOpts(interval=0,
                                                                               rotate=-45),
                                         axisline_opts=echarts_opts.AxisLineOpts(
                                             symbol=['none', 'arrow']),
                                         axispointer_opts=echarts_opts.AxisPointerOpts(
                                             is_show=True)

                                         ),
        )
             )

    page = Page(page_title=title)
    page.add(heatmap, box, bar, line, bar1, line1)
    if FLAGS.train:
        page_path = os.path.join(config_params["save_path"], "kernerl_echarts.html")
    else:
        page_path = os.path.join(config_params["save_path"], "statistic_kernerl_echarts.html")
    while os.path.exists(page_path):
        if page_path[-6].isdigit():
            page_path = page_path[:-6] + str(int(page_path[-6]) + 1) + ".html"
        else:
            page_path = page_path[:-6] + "_1" + ".html"
    page.render(page_path)
    logger.info("统计参数分布情况,保存在%s" % (page_path))


def datadisplay_method(filename, config_params):
    numb_fig = 3
    with open(filename, "r") as f:
        lines = f.readlines()
        train_acc_ = []
        test_acc_ = []
        train_cost_ = []
        test_cost_ = []
        lr_ = []
        if FLAGS.crossgroup:
            train_cg_ = []
            test_cg_ = []
            numb_fig += 1
        if FLAGS.grouplasso:
            train_gl_ = []
            test_gl_ = []
            numb_fig += 1
        if FLAGS.regularizer:
            train_reg_ = []
            test_reg_ = []
            numb_fig += 1
        for line in lines:
            line = line.strip().split(",")
            train_acc_.append(float(line[0]))
            test_acc_.append(float(line[1]))
            train_cost_.append(float(line[2]))
            test_cost_.append(float(line[3]))
            lr_.append(float(line[-1]))
            index = 4
            if FLAGS.regularizer:
                train_reg_.append(float(line[index]))
                test_reg_.append(float(line[index + 1]))
                index += 2
            if FLAGS.crossgroup:
                train_cg_.append(float(line[index]))
                test_cg_.append(float(line[index + 1]))
                index += 2
            if FLAGS.grouplasso:
                train_gl_.append(float(line[index]))
                test_gl_.append(float(line[index]))

    # plt
    if numb_fig == 3:
        plotnumb = 131
    elif numb_fig == 4:
        plotnumb = 221
    else:
        plotnumb = 231
    acc_index, acc_max = findmax(test_acc_)
    train_acc_index, train_acc_max = findmax(train_acc_)
    title = "testacc:" + str(round(acc_max, 4))
    plt.figure(figsize=(10, 8))
    plt.subplot(plotnumb)
    plt.plot(range(1, len(train_acc_) + 1), train_acc_, label="train_acc")
    plt.plot(range(1, len(test_acc_) + 1), test_acc_, label="test_acc")
    plt.text(acc_index, acc_max, "%.4f" % acc_max, ha='center', va='bottom', fontsize=7)
    plt.text(train_acc_index, train_acc_max, "%.4f" % train_acc_max, ha='center', va='bottom',
             fontsize=7)
    plt.title("the accuracy of train and test ")
    plt.legend()
    plotnumb += 1
    plt.subplot(plotnumb)
    plt.plot(range(1, len(train_cost_) + 1), train_cost_, label="train_cost")
    plt.plot(range(1, len(test_cost_) + 1), test_cost_, label="test_cost")
    plt.title("cost of cross entropy ")
    plt.legend()
    plotnumb += 1
    plt.subplot(plotnumb)
    plt.plot(range(1, len(lr_) + 1), lr_, label="learning_rate")
    plt.title("learning rates ")
    plt.legend()
    plotnumb += 1
    if FLAGS.crossgroup:
        plt.subplot(plotnumb)
        plt.plot(range(1, len(train_cg_) + 1), train_cg_, label="train_cg")
        plt.plot(range(1, len(test_cg_) + 1), test_cg_, label="test_cg")
        plt.title("cross lasso")
        title += " cg para: " + str(config_params["crossgroup_para"]) + " "
        plt.legend()
        plotnumb += 1
    if FLAGS.grouplasso:
        plt.subplot(plotnumb)
        plt.plot(range(1, len(train_gl_) + 1), train_gl_, label="train_gl")
        plt.plot(range(1, len(test_gl_) + 1), test_gl_, label="test_gl")
        plt.title("group lasso")
        title += " gl para: " + str(config_params["grouplasso_para1"]) + "," + str(
            config_params["grouplasso_para2"]) + " "
        plt.legend()
        plotnumb += 1
    if FLAGS.regularizer:
        plt.subplot(plotnumb)
        plt.plot(range(1, len(train_reg_) + 1), train_reg_, label="train_reg")
        plt.plot(range(1, len(test_reg_) + 1), test_reg_, label="test_reg")
        str_reg = FLAGS.regularizer
        plt.title("change of " + str_reg + " regularizer")
        if str_reg == "L2":
            regtitle = config_params["L2para"]
        else:
            regtitle = config_params["L1para"]
        title += " reg para: " + str(regtitle)
        plt.legend()
        plotnumb += 1
    plt.tight_layout()
    plt.suptitle(title)
    plt.subplots_adjust(top=0.90)
    figsava_path = os.path.join(config_params["save_path"], "fig.jpg")
    while os.path.exists(figsava_path):
        if figsava_path[-5].isdigit():
            figsava_path = figsava_path[:-5] + str(int(figsava_path[-5]) + 1) + ".jpg"
        else:
            figsava_path = figsava_path[:-5] + "_1" + ".jpg"
    plt.savefig(figsava_path)
    plt.show()


def find_freeze(sess, md, x, y_, test_x, test_y, cross_entropy, accuracy,
                is_training,
                cg, gl, reg,
                keep_prob, sava_val_acc):
    weight_list_tensor = md.get_weitht()
    bias_list_tensor = md.get_bias()
    no_loss_weight = []
    no_loss_bias = []
    weight_actually = sess.run(md.filter())
    for i in range(len(weight_actually)):
        weight_actually[i] = weight_actually[i].astype(np.float64)

    for i in range(len(weight_actually)):
        temp_weight = sess.run(weight_list_tensor[i])
        temp_bias = sess.run(bias_list_tensor[i])
        l_shape = weight_list_tensor[i].get_shape().as_list()
        layer_th = 1
        _set_i = 1
        loss_flag = 0
        while layer_th > 0.0000001:
            bias_bool = weight_actually[i] >= layer_th
            weight_bool = np.tile(bias_bool, [l_shape[0], l_shape[1], l_shape[2], 1])
            sess.run(tf.assign(weight_list_tensor[i],
                               tf.where(weight_bool, weight_list_tensor[i],
                                        tf.zeros_like(weight_list_tensor[i]))))
            sess.run(tf.assign(bias_list_tensor[i],
                               tf.where(bias_bool, bias_list_tensor[i],
                                        tf.zeros_like(bias_list_tensor[i]))))
            cur_acc, _, _, _, _ = run_testing(sess, x, y_, test_x, test_y, cross_entropy, accuracy,
                                              is_training,
                                              cg, gl, reg,
                                              keep_prob)
            if cur_acc >= sava_val_acc:
                no_loss_weight.append(weight_bool)
                no_loss_bias.append(bias_bool)
                loss_flag = 1
                break
            else:
                if _set_i % 2 == 1:
                    layer_th /= 2
                else:
                    layer_th /= 5
                _set_i += 1
                sess.run(tf.assign(weight_list_tensor[i], temp_weight))
                sess.run(tf.assign(bias_list_tensor[i], temp_bias))
        if loss_flag == 0:
            no_loss_weight.append(np.ones_like(temp_weight))
            no_loss_bias.append(np.ones_like(temp_bias))
            sess.run(tf.assign(weight_list_tensor[i], temp_weight))
            sess.run(tf.assign(bias_list_tensor[i], temp_bias))

    finally_acc, _, _, _, _ = run_testing(sess, x, y_, test_x, test_y, cross_entropy, accuracy,
                                          is_training,
                                          cg, gl, reg,
                                          keep_prob)
    return no_loss_weight, no_loss_bias, finally_acc

def train_find_freeze(sess, md, x, y_, train_x, train_y, cross_entropy, accuracy,
                is_training,
                cg, gl, reg,
                keep_prob, sava_val_acc):
    weight_list_tensor = md.get_weitht()
    bias_list_tensor = md.get_bias()
    no_loss_weight = []
    no_loss_bias = []
    weight_actually = sess.run(md.filter())
    for i in range(len(weight_actually)):
        weight_actually[i] = weight_actually[i].astype(np.float64)

    for i in range(len(weight_actually)):
        temp_weight = sess.run(weight_list_tensor[i])
        temp_bias = sess.run(bias_list_tensor[i])
        l_shape = weight_list_tensor[i].get_shape().as_list()
        layer_th = 1
        _set_i = 1
        loss_flag = 0
        while layer_th > 0.0001:
            bias_bool = weight_actually[i] >= layer_th
            weight_bool = np.tile(bias_bool, [l_shape[0], l_shape[1], l_shape[2], 1])
            sess.run(tf.assign(weight_list_tensor[i],
                               tf.where(weight_bool, weight_list_tensor[i],
                                        tf.zeros_like(weight_list_tensor[i]))))
            sess.run(tf.assign(bias_list_tensor[i],
                               tf.where(bias_bool, bias_list_tensor[i],
                                        tf.zeros_like(bias_list_tensor[i]))))
            cur_acc, _, _, _, _ = run_training(sess, x, y_, train_x, train_y, cross_entropy, accuracy,
                                              is_training,
                                              cg, gl, reg,
                                              keep_prob)
            if cur_acc >= sava_val_acc:
                no_loss_weight.append(weight_bool)
                no_loss_bias.append(bias_bool)
                loss_flag = 1
                break
            else:
                if _set_i % 2 == 1:
                    layer_th /= 2
                else:
                    layer_th /= 5
                _set_i += 1
                sess.run(tf.assign(weight_list_tensor[i], temp_weight))
                sess.run(tf.assign(bias_list_tensor[i], temp_bias))
        if loss_flag == 0:
            no_loss_weight.append(np.ones_like(temp_weight))
            no_loss_bias.append(np.ones_like(temp_bias))
            sess.run(tf.assign(weight_list_tensor[i], temp_weight))
            sess.run(tf.assign(bias_list_tensor[i], temp_bias))

    finally_acc, _, _, _, _ = run_training(sess, x, y_, train_x, train_y, cross_entropy, accuracy,
                                          is_training,
                                          cg, gl, reg,
                                          keep_prob)
    return no_loss_weight, no_loss_bias, finally_acc


def train_mode(config_params, sess, global_step, train_x, train_y, test_x, test_y, train_ops, cross_entropy, accuracy,
               gl, cg, reg, zero_ops,
               x, y_, learning_rate, is_training, keep_prob, max_to_keep, saver, logger,l2_flag,md):
    data_time = strftime("%Y-%m-%d__%H-%M-%S", localtime())
    filename = os.path.join(config_params["save_path"], data_time + "data_record.txt")
    File = open(filename, 'a+')
    begin_acc = config_params["begin_acc"]
    batch_size = config_params["batch_size"]
    iterations = config_params["iterations"]
    if FLAGS.retrain:
        total = config_params["retrain_total"]
    elif FLAGS.partial_retrain:
        total = config_params["partial_retain_total"]
    else:
        total = config_params["total"]
    start = config_params["start"]
    temp_acc = 0
    temp_train_acc = 0
    k = 0
    weight_list = md.get_weitht()
    bias_list = md.get_bias()
    lr_step = 0
    max_acc = 0.0
    for ep in range(start, total + 1):
        if FLAGS.LR == "step":
            lr = learning_rate_schedule(ep, config_params)
        elif FLAGS.LR == "Ex":
            lr = sess.run(lr_exponential(ep, config_params, global_step))
            sess.run(tf.assign(global_step, ep - 1))
        if FLAGS.retrain:
            lr = larning_rate_retrain(ep, config_params)
        if FLAGS.partial_retrain:
            lr = learning_rate_partial_retrain(ep,config_params)
        use_flag = use_flag_schedule(ep, config_params)
        reg = use_flag * reg

        # if ep == 90 or ep == 130:
        #     no_loss_weight, no_loss_bias, finally_acc = find_freeze(sess, md, x, y_, test_x, test_y, cross_entropy,
        #                                                             accuracy,
        #                                                             is_training,
        #                                                             cg, gl, reg,
        #                                                             keep_prob, temp_acc)
        #     logger.info("冻结相应变量精度为%.4f" % (finally_acc))
        #     md.pass_weight_bias(no_loss_weight, no_loss_bias)
        #     train_ops = md.train_op
        #     zero_ops = md.zero_op
        #     global_vars = tf.global_variables()
        #
        #     is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        #     not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        #
        #     if len(not_initialized_vars):
        #         logger.info("没有初始化的变量:")
        #         logger.info([str(i.name) for i in not_initialized_vars])  # only for testing
        #         sess.run(tf.variables_initializer(not_initialized_vars))
        # if ep == 2:
        #     no_loss_weight, no_loss_bias, finally_acc = train_find_freeze(sess, md, x, y_, train_x, train_y,
        #                                                                   cross_entropy,
        #                                                                   accuracy,
        #                                                                   is_training,
        #                                                                   cg, gl, reg,
        #                                                                   keep_prob, temp_train_acc)
        #     logger.info("冻结相应变量精度为%.4f" % (finally_acc))
        #     md.pass_weight_bias(no_loss_weight, no_loss_bias)
        #     train_ops = md.train_op
        #     zero_ops = md.zero_op
        #     global_vars = tf.global_variables()
        #
        #     is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        #     not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        #
        #     if len(not_initialized_vars):
        #         logger.info("没有初始化的变量:")
        #         logger.info([str(i.name) for i in not_initialized_vars])  # only for testing
        #         sess.run(tf.variables_initializer(not_initialized_vars))
        # if FLAGS.flag:
        #     phase1 = config_params["lock_epoch"]["start"]
        #     phase2 = config_params["lock_epoch"]["end"]
        #     for i in range(len(phase1)):
        #         if ep >= phase1[i] and ep < phase2[i]:
        #             md.filter_flag() #计数
        #         if ep == phase2[i]:
        #             epoch = int(phase2[i]-phase1[i])
        #             count_ = md.filter_count   ##得到相应的计数
        #             for j in range(len(count_)):
        #                 count_1 = tf.greater(count_[j], epoch-1)
        #                 l_shape = weight_list[j].get_shape().as_list()
        #                 bias_temp = tf.where(count_1, bias_list[j], tf.zeros_like(bias_list[j]))
        #                 sess.run(tf.assign(bias_list[j], bias_temp))  ##偏置赋值
        #                 count_1 = tf.expand_dims(count_1,0)
        #                 count_1 = tf.expand_dims(count_1, 0)
        #                 count_1 = tf.expand_dims(count_1, 0)
        #                 count_1 = tf.tile(count_1, [l_shape[0], l_shape[1], l_shape[2], 1])
        #                 weight_temp = tf.where(count_1,weight_list[j],tf.zeros_like(weight_list[j]))
        #                 sess.run(tf.assign(weight_list[j], weight_temp))  ##过滤器赋值
        #
        #             cur_acc, _, _, _, _ = run_testing(sess, x, y_, test_x, test_y, cross_entropy, accuracy,
        #                                               is_training,
        #                                               cg, gl, reg,
        #                                               keep_prob)
        #             logger.info("the %d look lock ,the acc is %.4f"%(ep,cur_acc))
        #             md.default_count()

        if FLAGS.flag:
            if ep >=60 and ep <=210:
                if k == 10:
                    count_ = md.filter_count
                    for j in range(len(count_)):
                        count_1 = tf.greater_equal(count_[j], 5)
                        l_shape = weight_list[j].get_shape().as_list()
                        bias_temp = tf.where(count_1, bias_list[j], tf.zeros_like(bias_list[j]))
                        sess.run(tf.assign(bias_list[j], bias_temp))  ##偏置赋值
                        count_1 = tf.expand_dims(count_1,0)
                        count_1 = tf.expand_dims(count_1, 0)
                        count_1 = tf.expand_dims(count_1, 0)
                        count_1 = tf.tile(count_1, [l_shape[0], l_shape[1], l_shape[2], 1])
                        weight_temp = tf.where(count_1,weight_list[j],tf.zeros_like(weight_list[j]))
                        sess.run(tf.assign(weight_list[j], weight_temp))  ##过滤器赋值

                    # cur_acc, _, _, _, _ = run_testing(sess, x, y_, test_x, test_y, cross_entropy, accuracy,
                    #                                   is_training,
                    #                                   cg, gl, reg,
                    #                                   keep_prob)
                    # logger.info("the %d look lock ,the acc is %.4f" % (ep, cur_acc))

                    md.default_count()
                    k = 0
                else:
                    md.filter_flag()  # 计数
                    k += 1
        if FLAGS.lock:
            if ep in config_params["lock"]:
                no_loss_weight, no_loss_bias, finally_acc = train_find_freeze(sess, md, x, y_, train_x, train_y,
                                                                              cross_entropy,
                                                                              accuracy,
                                                                              is_training,
                                                                              cg, gl, reg,
                                                                              keep_prob, temp_train_acc)
                logger.info("冻结相应变量精度为%.4f" % (finally_acc))
                md.pass_weight_bias(no_loss_weight, no_loss_bias)
                train_ops = md.train_op
                zero_ops = md.zero_op
                global_vars = tf.global_variables()

                is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
                not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

                if len(not_initialized_vars):
                    logger.info("没有初始化的变量:")
                    logger.info([str(i.name) for i in not_initialized_vars])  # only for testing
                    sess.run(tf.variables_initializer(not_initialized_vars))
        pre_index = 0
        train_acc = 0.0
        train_cost = 0.0
        train_gl = 0.0
        train_cg = 0.0
        train_reg = 0.0
        start_time = time.time()
        for it in range(1, iterations + 1):
            lr_step += 1
            if FLAGS.LR == "cycle":
                lr = cycle_larning_rate(lr_step, config_params)
                if ep > 60 or ep <= 63:
                    lr = 0.0001
                elif ep > 63 or ep <= 66:
                    lr = 0.00001
                elif ep >66 :
                    lr = 0.000001
            batch_x = train_x[pre_index:pre_index + batch_size]
            batch_y = train_y[pre_index:pre_index + batch_size]
            batch_x = loadcifardata.data_augmentation(batch_x)

            # sess.run(zero_ops)
            _, batch_cost, batch_acc, batch_gl, batch_cg, batch_reg, _ = sess.run(
                [train_ops, cross_entropy, accuracy, gl, cg, reg, zero_ops],
                feed_dict={x: batch_x, y_: batch_y, learning_rate: lr, is_training: True, keep_prob: 0.5,l2_flag:use_flag})
            train_cost += batch_cost
            train_acc += batch_acc
            train_cg += batch_cg
            train_gl += batch_gl
            train_reg += batch_reg
            pre_index += batch_size
            # 单次训练结束
            if it == iterations:
                train_cost /= iterations
                train_acc /= iterations
                train_cg /= iterations
                train_gl /= iterations
                train_reg /= iterations
                val_acc, val_cost, val_reg, val_cg, val_gl = run_testing(sess, x, y_, test_x, test_y, cross_entropy,
                                                                         accuracy, is_training, cg, gl, reg,
                                                                         keep_prob)
                temp_train_acc = train_acc
                temp_acc = val_acc
                File.write(str(train_acc) + ",")
                File.write(str(val_acc) + ",")
                File.write(str(train_cost) + ",")
                File.write(str(val_cost) + ",")
                # if (begin_acc < val_acc and max_acc < val_acc) :
                #     saver.save(sess, config_params["model_save_path"], ep)
                #     max_acc = val_acc

                if max_to_keep == 1:
                    if begin_acc < val_acc and max_acc < val_acc:
                        saver.save(sess, config_params["model_save_path"], ep)
                        max_acc = val_acc
                    # if ep == total:
                    #     saver.save(sess, config_params["model_save_path"], ep)
                    #     max_acc = val_acc
                else:
                    if (begin_acc < val_acc and max_acc < val_acc) or ep == config_params["total"]:
                        saver.save(sess, config_params["model_save_path"], ep)
                        if max_acc < val_acc:
                            max_acc = val_acc
                if total == 1:
                    saver.save(sess, config_params["model_save_path"], ep)

                weight_list = tf.get_collection("weight")

                ##sava gl value


                sava_file_f = os.path.join(config_params["value_path"], "f"+ str(ep))
                sava_file_c = os.path.join(config_params["value_path"], "c"+ str(ep))
                ff ,cc = md.filter_tensor,md.channel_tensor
                np.save(sava_file_f, sess.run(ff))
                np.save(sava_file_c, sess.run(cc))


                # if ep == 1:
                #     cg_extend_up =cg_extend
                # else:
                #     sava = []
                #     for ee in range(len(cg_extend)):  ##层
                #         sava.append([])
                #         for cc in range(len(cg_extend[ee])):  ##个数
                #             rate = tf.reduce_sum(cg_extend_up[ee][cc]*cg_extend[ee][cc])/(tf.norm(cg_extend_up[ee][cc],ord=2)
                #                                                                           *tf.norm(cg_extend[ee][cc],ord=2))
                #             sava[ee].append(rate)
                #     cg_extend_up = cg_extend
                #     np.save(sava_file, sess.run(sava))







                # if FLAGS.grouplasso or FLAGS.crossgroup:
                #     filter_value = md.gl_filter_value
                #     channel_value = md.gl_channel_value
                #     numpy_filter_value,numpy_channel_value = sess.run([filter_value,channel_value])
                #     if FLAGS.grouplasso :
                #         sava_name = "gl"
                #     else:
                #         sava_name = "cg"
                #     sava_file = os.path.join(config_params["value_path"],sava_name+str(ep))
                #     np.savez(sava_file,numpy_filter_value,numpy_channel_value)

                # if FLAGS.crossgroup:
                #     filter_value = md.gl_filter_value
                #     channel_value = md.gl_channel_value
                #     numpy_filter_value, numpy_channel_value = sess.run([filter_value, channel_value])
                #     sava_file = os.path.join(config_params["value_path"], "gl" + str(ep))
                #     np.savez(sava_file, numpy_filter_value="filter", numpy_channel_value="channel")
                #     filter_value = md.cg_filter_value
                #     numpy_filter_value = sess.run(filter_value)
                #     sava_file = os.path.join(config_params["value_path"], "cg" + str(ep))
                #     np.save(sava_file,numpy_filter_value)
                print("iteration: %d/%d, cost_time: %ds, train_cost: %.4f, "
                      "train_acc: %.4f, test_cost: %.4f, test_acc: %.4f,learning_rates:%.8f"
                      % (
                          ep, total, int(time.time() - start_time), train_cost, train_acc, val_cost, val_acc, lr),
                      end=" ")
                if FLAGS.regularizer:
                    print("train_reg%.4f, test_reg%.4f" % (train_reg, val_reg), end=" ")
                    File.write(str(train_reg) + ",")
                    File.write(str(val_reg) + ",")
                if FLAGS.crossgroup:
                    print("train_cg%.4f, test_cg%.4f" % (train_cg, val_cg), end=" ")
                    File.write(str(train_cg) + ",")
                    File.write(str(val_cg) + ",")
                if FLAGS.grouplasso:
                    print("train_gl%.4f ,test_gl%.4f" % (train_gl, val_gl), end=" ")
                    File.write(str(train_gl) + ",")
                    File.write(str(val_gl) + ",")
                print(end="\n")
                File.write(str(lr) + "\n")
                File.flush()
    File.close()
    logger.info('最大精度为%.4f ' % (max_acc))
    logger.info("train done! ")
    return filename

def picshow(x_p, acc_p, url):
    # plt.rcParams['font.sans-serif'] = 'SimHei'  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    color = ["#FF4500", "#FFFF00", "#FF00FF", "#8B0000", "#030303", "#00F5FF"]
    for i in range(len(acc_p) - 1):
        la = "conv_" + str(i + 1)
        if i < 6:
            plt.plot(x_p, acc_p[i],label=la, color=color[i], linestyle="--",marker='o')
            #plt.plot(x_p, acc_p[i], 'o')
        else:
            plt.plot(x_p, acc_p[i],label=la, color=color[i - 6], linestyle="-",marker='*')
            #plt.plot(x_p, acc_p[i], 'o')

    plt.plot(x_p, acc_p[12],label="layer13", color="#104E8B", linestyle=":",marker="+")
    #plt.plot(x_p, acc_p[12], 'o')
    plt.xlabel('pruning rate(%)')
    plt.ylabel('accuracy(%)')
    plt.title("Cifar-10 Vgg16 Baseline")
    plt.legend()
    plt.savefig(url,dpi=600)

def cutmodel(weight,sava_acc,sess, x, y_, test_x, test_y,cross_entropy,
             accuracy, is_training, cg, gl, reg,keep_prob):
    acc = []
    x_p = []
    for i in range(len(weight)):
        acc.append([])
        acc[i].append(sava_acc)
        weight_temp = sess.run(weight[i])
        weight_shape = weight_temp.shape
        weight_op_ = weight_temp.copy()
        weight_zero = np.zeros_like(weight_op_)
        weight_op = np.abs(weight_op_)
        weight_op = np.sum(weight_op,axis=(0,1,2))
        index = np.argsort(weight_op)
        for p in range(1,11,1):
            value = weight_op[index[len(weight_op)*p//10-1]]
            value_f = value >= weight_op
            value_f = np.tile(value_f,(weight_shape[0],weight_shape[1],weight_shape[2],1))
            weight_assign = np.where(value_f,weight_zero,weight_op_)
            sess.run(tf.assign(weight[i],weight_assign))
            val_acc, val_cost, val_reg, val_cg, val_gl = run_testing(sess, x, y_, test_x, test_y,
                                                                          cross_entropy,
                                                                          accuracy, is_training, cg, gl, reg,
                                                                          keep_prob)
            acc[i].append(val_acc)
        sess.run(tf.assign(weight[i], weight_temp))
    for i in range(len(acc[0])):
        x_p.append(i*10)
    return acc,x_p

def load_train_data(train_location):
    train_dict = sio.loadmat(train_location)
    X = np.asarray(train_dict['X'])

    X_train = []
    for i in range(X.shape[3]):
        X_train.append(X[:,:,:,i])
    X_train = np.asarray(X_train)

    Y_train = train_dict['y']
    for i in range(len(Y_train)):
        if Y_train[i]%10 == 0:
            Y_train[i] = 0
    Y_train = np.array([[float(i == label) for i in range(10)] for label in Y_train])
    # Y_train = to_categorical(Y_train,10)

    return (X_train,Y_train)

def load_test_data(test_location):
    test_dict = sio.loadmat(test_location)
    X = np.asarray(test_dict['X'])

    X_test = []
    for i in range(X.shape[3]):
        X_test.append(X[:,:,:,i])
    X_test = np.asarray(X_test)

    Y_test = test_dict['y']
    for i in range(len(Y_test)):
        if Y_test[i]%10 == 0:
            Y_test[i] = 0
    Y_test = np.array([[float(i == label) for i in range(10)] for label in Y_test])
    # Y_test = to_categorical(Y_test,10)
    return (X_test,Y_test)

def main(argv):
    ## judge whether the relevant parament correct
    flag_list = ["model=", "restore_path=", "config_file=", "regularizer=", "crossgroup=", "optimizer=",
                 "freeze_weight=", "statistic=", "datadisplay=", "store_weight=", "train=", "dpfilename=",
                 "reverse=", "LR=", "retrain=","partial_retrain=","flag=","lookrate=","lock","cg_value=",
                 "gl_1=", "gl_2=","epoch="]
    try:
        opts, args = getopt.getopt(argv[1:], "", flag_list)
    except getopt.GetoptError:
        raise ValueError("Must set correct parameter")
    if not FLAGS.config_file:
        raise ValueError("Must set --config_file to configuration file")
    else:
        with open(FLAGS.config_file, 'r') as fi:
            config_params = json.load(fi)
    model_list = ["Vgg", "ResNet","Lenet"]

    if FLAGS.model not in model_list:
        ValueError("the specify model hasn't been implemented,suggested use: Vgg, ResNet")
    reg_list = ["L1", "L2"]
    if FLAGS.regularizer not in reg_list:
        ValueError("the specify regluarizer hasn't been implemented,suggested use: L1, L2")
    optimizer_list = ["Moment", "adam", "gd"]
    if FLAGS.optimizer not in optimizer_list:
        ValueError("the specify optimizer hasn't been implemented,suggested use: 'Moment', 'adam','gd'")

    ##config 参数兑现
    max_to_keep = config_params["max_to_keep"]  # 最大保存数

    if FLAGS.cg_value != 0.0:
        config_params["crossgroup_para"] = FLAGS.cg_value
    if FLAGS.gl_1 != 0.0:
        config_params["grouplasso_para1"] = FLAGS.gl_1
    if FLAGS.gl_2 != 0.0:
        config_params["grouplasso_para2"] = FLAGS.gl_2
    if FLAGS.epoch != 0.0:
        config_params["total"] = int(FLAGS.epoch)
        # get logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('shark')
    logger.setLevel(logging.INFO)

    # 数据准备
    if config_params["is_cifar_data"] == 1:
        raw_data = loadcifardata.Cifar(config_params["data_path"])
        train_x, train_y, test_x, test_y = raw_data.prepare_data()
        print("use cifar data")
    else:
        train_x, train_y = load_train_data(config_params["train_path"])
        test_x, test_y = load_test_data(config_params["test_path"])
        print("use SVHN data")
    train_x, test_x = loadcifardata.data_preprocessing(train_x, test_x)

    x = tf.placeholder(tf.float32,
                       [None, config_params["image_size"], config_params["image_size"], config_params["image_channel"]])
    y_ = tf.placeholder(tf.float32, [None, config_params["class_num"]])
    learning_rate = tf.placeholder(tf.float32)
    l2_flag = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    if FLAGS.model == "ResNet":
        model = ResNetModel(x, config_params["num_residual_blocks"], config_params["num_filter_base"],
                            config_params["class_num"], is_training)
    elif FLAGS.model == "Vgg" or FLAGS.model == "vgg":
        model = VggModel(x, keep_prob, is_training, config_params)
    elif FLAGS.model == "Lenet" or FLAGS.model == "LeNet":
        model = Lenet_Model(x, keep_prob, is_training, config_params)
    else:
        raise ValueError("the other model doesn't come true")

    outputs = model.output  # resnet模型的输出
    md = ModelDeploy(y_, outputs, config_params, learning_rate,l2_flag)
    cross_entropy = md.cost
    cg = md.crossgroup
    reg = md.regularization
    gl = md.grouplasso
    weight_list = tf.get_collection("weight")

    accuracy = md.accuracy
    if FLAGS.partial_retrain == True:
        if os.path.exists(FLAGS.restore_path):
            with tf.Session() as sess:
                cur_path = FLAGS.restore_path
                config_params["save_path"] = cur_path + "partial_restore"
                while os.path.exists(config_params["save_path"]):
                    if config_params["save_path"][-1].isdigit():
                        config_params["save_path"] = config_params["save_path"][:-1] + str(
                            int(config_params["save_path"][-1]) + 1)
                    else:
                        config_params["save_path"] = config_params["save_path"] + "_1"
                os.makedirs(config_params['save_path'])
                log_file = os.path.join(config_params['save_path'], 'running.log')
                while os.path.exists(log_file):
                    if log_file[-5].isdigit():
                        log_file = log_file[:-5] + str(int(log_file[-5]) + 1) + ".log"
                    else:
                        log_file = log_file[:-5] + "_1" + ".log"

                logger.addHandler(logging.FileHandler(log_file))  # 记录
                config_params['model_save_path'] = os.path.join(config_params['save_path'], "re")
                logger.info('configurations in file:\n %s \n', config_params)
                train_ops = md.train_op
                zero_ops = md.zero_op
                global_vars = tf.global_variables()

                # variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['conv12'])
                saver = tf.train.Saver(max_to_keep=max_to_keep)  # 保存模型
                ckpt = tf.train.get_checkpoint_state(FLAGS.restore_path)
                max_path = 0 - max_to_keep
                saver.restore(sess, ckpt.all_model_checkpoint_paths[max_path])

                is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
                not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
                if len(not_initialized_vars):
                    logger.info("没有初始化的变量如下:")
                    logger.info([str(i.name) for i in not_initialized_vars])  # only for testing
                    sess.run(tf.variables_initializer(not_initialized_vars))
                temp_var  = tf.get_variable('a',shape=[3,3,512,512],initializer=tf.random_normal_initializer(stddev=1))
                sess.run(tf.variables_initializer([temp_var]))
                sess.run(tf.assign(tf.get_collection("weight")[11],temp_var))

                sava_val_acc, val_cost, val_reg, val_cg, val_gl = run_testing(sess, x, y_, test_x, test_y,
                                                                              cross_entropy,
                                                                              accuracy, is_training, cg, gl, reg,
                                                                              keep_prob)
                logger.info("恢复时精度为%.4f" % (sava_val_acc))

                # sess.run(zero_ops)
                # freeze_acc, _, _, _, _ = run_testing(sess, x, y_, test_x, test_y, cross_entropy, accuracy,
                #                                   is_training,
                #                                   cg, gl, reg,
                #                                   keep_prob)
                # logger.info("freeze_acc is %.4f"%(freeze_acc))
                # saver = tf.train.Saver(max_to_keep=max_to_keep)  # 保存模型
                # ckpt = tf.train.get_checkpoint_state(FLAGS.restore_path)
                # max_path = 0 - max_to_keep
                # saver.restore(sess, ckpt.all_model_checkpoint_paths[max_path])

                ## train
                global_step = tf.Variable(0, trainable=False)

                # md1 = ModelDeploy(y_, outputs, config_params, learning_rate,no_loss_weight,no_loss_bias)
                # cross_entropy = md1.cost
                # cg = md1.crossgroup
                # reg = md1.regularization
                # gl = md1.grouplasso
                # accuracy = md1.accuracy

                # saver = tf.train.Saver(max_to_keep=max_to_keep)  # 保存模型
                # ckpt = tf.train.get_checkpoint_state(FLAGS.restore_path)
                # max_path = 0 - max_to_keep
                # saver.restore(sess, ckpt.all_model_checkpoint_paths[max_path])

                filename = train_mode(config_params, sess, global_step, train_x, train_y, test_x, test_y, train_ops,
                                      cross_entropy,
                                      accuracy, gl, cg, reg, zero_ops,
                                      x, y_, learning_rate, is_training, keep_prob, max_to_keep, saver, logger,l2_flag,md)

                # ckpt = tf.train.get_checkpoint_state(config_params['save_path'] + "/")
                # max_path = 0 - max_to_keep
                # saver.restore(sess, ckpt.all_model_checkpoint_paths[max_path])
                ## 取得restore 最大变量
                ckpt = tf.train.get_checkpoint_state(config_params['save_path'] + "/")
                max_path = 0 - max_to_keep
                saver.restore(sess, ckpt.all_model_checkpoint_paths[max_path])
                if FLAGS.store_weight:
                    store_weight_method(config_params, sess, logger)
                if FLAGS.statistic:
                    title = FLAGS.model
                    if FLAGS.crossgroup:
                        title += " " + "cg_" + str(config_params["crossgroup_para"])
                    elif FLAGS.grouplasso:
                        title += " " + "gl_" + str(config_params["grouplasso_para1"])
                    elif FLAGS.regularizer:
                        title += " " + "reg" + str(config_params["L2para"])
                    else:
                        title += " " + "orignal"
                    statistic_method(sess, x, y_, test_x, test_y, cross_entropy, accuracy,
                                     is_training, cg, gl, reg, keep_prob, logger, md, model, config_params,
                                     max_to_keep, saver, log_file, title)
                if FLAGS.datadisplay:
                    datadisplay_method(filename, config_params)
        else:
            raise IOError('%s not exist!' % FLAGS.restore_path)

        return

    if FLAGS.retrain == True:
        if os.path.exists(FLAGS.restore_path):
            with tf.Session() as sess:
                cur_path = FLAGS.restore_path
                config_params["save_path"] = cur_path + "restore"
                while os.path.exists(config_params["save_path"]):
                    if config_params["save_path"][-1].isdigit():
                        config_params["save_path"] = config_params["save_path"][:-1] + str(
                            int(config_params["save_path"][-1]) + 1)
                    else:
                        config_params["save_path"] = config_params["save_path"] + "_1"
                os.makedirs(config_params['save_path'])
                log_file = os.path.join(config_params['save_path'], 'running.log')
                while os.path.exists(log_file):
                    if log_file[-5].isdigit():
                        log_file = log_file[:-5] + str(int(log_file[-5]) + 1) + ".log"
                    else:
                        log_file = log_file[:-5] + "_1" + ".log"

                logger.addHandler(logging.FileHandler(log_file))  # 记录
                config_params['model_save_path'] = os.path.join(config_params['save_path'], "re")
                logger.info('configurations in file:\n %s \n', config_params)
                saver = tf.train.Saver(max_to_keep=max_to_keep)  # 保存模型
                ckpt = tf.train.get_checkpoint_state(FLAGS.restore_path)
                max_path = 0 - max_to_keep
                max_path = -1
                saver.restore(sess, ckpt.all_model_checkpoint_paths[max_path])

                sava_val_acc, val_cost, val_reg, val_cg, val_gl = run_testing(sess, x, y_, test_x, test_y,
                                                                              cross_entropy,
                                                                              accuracy, is_training, cg, gl, reg,
                                                                              keep_prob)
                logger.info("恢复时精度为%.4f" % (sava_val_acc))
                no_loss_weight, no_loss_bias, finally_acc = find_freeze(sess, md, x, y_, test_x, test_y, cross_entropy,
                                                                        accuracy,
                                                                        is_training,
                                                                        cg, gl, reg,
                                                                        keep_prob, sava_val_acc)
                logger.info("冻结相应变量精度为%.4f" % (finally_acc))
                md.pass_weight_bias(no_loss_weight, no_loss_bias)
                train_ops = md.train_op
                zero_ops = md.zero_op
                # sess.run(zero_ops)
                # freeze_acc, _, _, _, _ = run_testing(sess, x, y_, test_x, test_y, cross_entropy, accuracy,
                #                                   is_training,
                #                                   cg, gl, reg,
                #                                   keep_prob)
                # logger.info("freeze_acc is %.4f"%(freeze_acc))
                # saver = tf.train.Saver(max_to_keep=max_to_keep)  # 保存模型
                # ckpt = tf.train.get_checkpoint_state(FLAGS.restore_path)
                # max_path = 0 - max_to_keep
                # saver.restore(sess, ckpt.all_model_checkpoint_paths[max_path])

                ## train
                global_step = tf.Variable(0, trainable=False)

                # md1 = ModelDeploy(y_, outputs, config_params, learning_rate,no_loss_weight,no_loss_bias)
                # cross_entropy = md1.cost
                # cg = md1.crossgroup
                # reg = md1.regularization
                # gl = md1.grouplasso
                # accuracy = md1.accuracy

                # saver = tf.train.Saver(max_to_keep=max_to_keep)  # 保存模型
                # ckpt = tf.train.get_checkpoint_state(FLAGS.restore_path)
                # max_path = 0 - max_to_keep
                # saver.restore(sess, ckpt.all_model_checkpoint_paths[max_path])

                global_vars = tf.global_variables()
                is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
                not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

                if len(not_initialized_vars):
                    logger.info("没有初始化的变量:")
                    logger.info([str(i.name) for i in not_initialized_vars])  # only for testing
                    sess.run(tf.variables_initializer(not_initialized_vars))

                filename = train_mode(config_params, sess, global_step, train_x, train_y, test_x, test_y, train_ops,
                                      cross_entropy,
                                      accuracy, gl, cg, reg, zero_ops,
                                      x, y_, learning_rate, is_training, keep_prob, max_to_keep, saver, logger,l2_flag,md)

                # ckpt = tf.train.get_checkpoint_state(config_params['save_path'] + "/")
                # max_path = 0 - max_to_keep
                # saver.restore(sess, ckpt.all_model_checkpoint_paths[max_path])
                ## 取得restore 最大变量
                ckpt = tf.train.get_checkpoint_state(config_params['save_path'] + "/")
                max_path = 0 - max_to_keep
                saver.restore(sess, ckpt.all_model_checkpoint_paths[max_path])
                if FLAGS.store_weight:
                    store_weight_method(config_params, sess, logger)
                if FLAGS.statistic:
                    title = FLAGS.model
                    if FLAGS.crossgroup:
                        title += " " + "cg_" + str(config_params["crossgroup_para"])
                    elif FLAGS.grouplasso:
                        title += " " + "gl_" + str(config_params["grouplasso_para1"])
                    elif FLAGS.regularizer:
                        title += " " + "reg" + str(config_params["L2para"])
                    else:
                        title += " " + "orignal"
                    statistic_method(sess, x, y_, test_x, test_y, cross_entropy, accuracy,
                                     is_training, cg, gl, reg, keep_prob, logger, md, model, config_params,
                                     max_to_keep, saver, log_file, title)
                if FLAGS.datadisplay:
                    datadisplay_method(filename, config_params)
        else:
            raise IOError('%s not exist!' % FLAGS.restore_path)

        return

    if FLAGS.train == False:

        if os.path.exists(FLAGS.restore_path):
            if FLAGS.datadisplay:
                if os.path.exists(FLAGS.dpfilename):
                    filename = FLAGS.dpfilename
                else:
                    raise ValueError("data display path not exit")
            with tf.Session() as sess:
                cur_path = FLAGS.restore_path
                config_params["save_path"] = cur_path[:-1]
                log_file = os.path.join(config_params['save_path'], 'statistic_running.log')
                while os.path.exists(log_file):
                    if log_file[-5].isdigit():
                        log_file = log_file[:-5] + str(int(log_file[-5]) + 1) + ".log"
                    else:
                        log_file = log_file[:-5] + "_1" + ".log"
                logger.addHandler(logging.FileHandler(log_file))  # 记录
                logger.info('configurations in file:\n %s \n', config_params)
                logger.info("验证已训练好的数据数据")
                # ckpt = tf.train.latest_checkpoint(FLAGS.restore_path)
                # saver.restore(sess, ckpt)
                saver = tf.train.Saver(max_to_keep=max_to_keep)  # 保存模型
                ckpt = tf.train.get_checkpoint_state(FLAGS.restore_path)
                max_path = 0 - max_to_keep
                saver.restore(sess, ckpt.all_model_checkpoint_paths[max_path])

                if FLAGS.lookrate:
                    sava_val_acc, val_cost, val_reg, val_cg, val_gl = run_testing(sess, x, y_, test_x, test_y,
                                                                                  cross_entropy,
                                                                                  accuracy, is_training, cg, gl, reg,
                                                                                  keep_prob)
                    logger.info("模型的识别率为%.4f" % (sava_val_acc))
                    weight = tf.get_collection("weight")
                    acc, x_p= cutmodel(weight, sava_val_acc, sess, x, y_, test_x, test_y, cross_entropy,
                             accuracy, is_training, cg, gl, reg, keep_prob)
                    picshow(x_p,acc,"./acc_pruning_rate.tif")
                    return

                if FLAGS.store_weight:
                    store_weight_method(config_params, sess, logger)
                if FLAGS.statistic:
                    title = FLAGS.model
                    if FLAGS.crossgroup:
                        title += " " + "cg_" + str(config_params["crossgroup_para"])
                    elif FLAGS.grouplasso:
                        title += " " + "gl_" + str(config_params["grouplasso_para1"])
                    elif FLAGS.regularizer:
                        title += " " + "reg" + str(config_params["L2para"])
                    else:
                        title += " " + "orignal"

                    statistic_method(sess, x, y_, test_x, test_y, cross_entropy, accuracy,
                                     is_training, cg, gl, reg, keep_prob, logger, md, model, config_params,
                                     max_to_keep, saver, log_file, title)
                if FLAGS.datadisplay:
                    datadisplay_method(filename, config_params)
                    # filename = "./resnet_model/ori/2020-01-13__09-20-45data_record.txt"
        else:
            raise IOError('%s not exist!' % FLAGS.restore_path)

        return

    # saving path
    if FLAGS.crossgroup:
        subfolder_name = "cg"
    elif FLAGS.grouplasso:
        subfolder_name = "gl"
    elif FLAGS.regularizer:
        subfolder_name = "reg"
    else:
        subfolder_name = "ori"
    config_params["save_path"] = FLAGS.model + "_model"
    config_params['save_path'] = os.path.join(config_params['save_path'], subfolder_name)
    while os.path.exists(config_params['save_path']):
        if config_params['save_path'][-1].isdigit():
            path_split = config_params['save_path'].split("_")
            config_params['save_path'] = path_split[0] + "_" + path_split[1] + "_" + str(int(path_split[2]) + 1)

        else:
            config_params['save_path'] = config_params['save_path'] + "_1"
    os.makedirs(config_params['save_path'])
    config_params['value_path'] = os.path.join(config_params["save_path"], "lookvalue")
    os.makedirs(config_params['value_path'])
    config_params['model_save_path'] = os.path.join(config_params['save_path'], subfolder_name)

    log_file = os.path.join(config_params['save_path'], 'running.log')
    logger.addHandler(logging.FileHandler(log_file))
    logger.info('configurations in file:\n %s \n', config_params)
    logger.info('tf FLAGS configurations:\n')
    # for name,value in FLAGS.__flags.i
    for key in FLAGS.flag_values_dict():
        if FLAGS[key].value:
            logger.info(str(key) + " : " + str(FLAGS[key].value))

    # logger.info('tf.FLAGS:\n %s \n', FLAGS)

    with tf.Session() as sess:
        global_step = tf.Variable(0, trainable=False)

        train_ops = md.train_op
        zero_ops = md.zero_op
        # sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=max_to_keep)  # 保存模型

        if not FLAGS.restore_path:
            sess.run(tf.global_variables_initializer())
        else:
            if os.path.exists(FLAGS.restore_path):
                ckpt = tf.train.latest_checkpoint(FLAGS.restore_path)
                saver.restore(sess, ckpt)
            else:
                raise IOError('%s not exist!' % FLAGS.restore_path)

        # for var in tf.trainable_variables():
        #     print(var.name)
        # print("-----------------------")
        # for var in tf.global_variables():
        #     print(var.name)

        filename = train_mode(config_params, sess, global_step, train_x, train_y, test_x, test_y, train_ops,
                              cross_entropy,
                              accuracy, gl, cg, reg, zero_ops, x, y_, learning_rate, is_training, keep_prob,
                              max_to_keep, saver, logger,l2_flag,md)
        ckpt = tf.train.get_checkpoint_state(config_params['save_path'] + "/")
        max_path = 0 - max_to_keep
        saver.restore(sess, ckpt.all_model_checkpoint_paths[max_path])

        if FLAGS.store_weight:
            store_weight_method(config_params, sess, logger)
        if FLAGS.statistic:
            title = FLAGS.model
            if FLAGS.crossgroup:
                title += " " + "cg_" + str(config_params["crossgroup_para"])
            elif FLAGS.grouplasso:
                title += " " + "gl_" + str(config_params["grouplasso_para1"])
            elif FLAGS.regularizer:
                title += " " + "reg" + str(config_params["L2para"])
            else:
                title += " " + "orignal"
            statistic_method(sess, x, y_, test_x, test_y, cross_entropy, accuracy,
                             is_training, cg, gl, reg, keep_prob, logger, md, model, config_params,
                             max_to_keep, saver, log_file, title)
        if FLAGS.datadisplay:
            datadisplay_method(filename, config_params)


if __name__ == "__main__":
    tf.app.run()
