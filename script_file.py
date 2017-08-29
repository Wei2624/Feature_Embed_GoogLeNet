import h5py
import numpy as np
import sample_generator as sg
import tensorflow as tf
import scipy.io

alpha = 0.01


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def fc_layer(x, input_dim, output_dim, flat=False, linear=False):
    W_fc = weight_variable([input_dim, output_dim])
    b_fc = bias_variable([output_dim])
    if flat: x = tf.reshape(x, [-1, input_dim])
    h_fc = tf.add(tf.matmul(x, W_fc), b_fc)
    if linear: return h_fc
    return tf.maximum(alpha * h_fc, h_fc)
#start to construct the model

class CNN_Triplet_Metric(object):
    def __init__(self,sess):
        self.img_a = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.img_p = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.img_n = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.sess = sess

        image_mean = scipy.io.loadmat('image_mean.mat')
        image_mean = image_mean['image_mean']



        # reading matlab v7.3 file using h5py. it has struct with img as a member
        with h5py.File("training_images_crop15_square256.mat") as f:
            img_data_train = [f[element[0]][:] for element in f['training_images/img']]
            class_id_train = [f[element[0]][:] for element in f['training_images/class_id']]

        img_data_train = np.float32(np.asarray(img_data_train))
        class_id_train = np.asarray(class_id_train)
        img_data_train = np.transpose(img_data_train, (0, 2, 3, 1))
        class_label = class_id_train[:, 0, 0]

        for i in range(len(class_label)-1):
            img_data_train[i,:,:,:] -= np.float32(image_mean)

        index_a, index_p, index_n = sg.generate_triplet(class_label, 120)

        with tf.variable_scope("") as scope:
            a_output = self.CNN_Metric_Model(self.img_a, True)
            scope.reuse_variables()
            p_output = self.CNN_Metric_Model(self.img_p, False)
            scope.reuse_variables()
            n_output = self.CNN_Metric_Model(self.img_n, False)

        loss,_,_ ,_= self.triplet_loss([1.0,1.0,1.0],a_output,p_output,n_output)
        test = self.sess.run([loss], feed_dict={self.img_a: img_data_train[index_a, :, :, :],
                                         self.img_p: img_data_train[index_p, :, :, :],
                                         self.img_n: img_data_train[index_n, :, :, :]})
        print test[0]

    def CNN_Metric_Model(self,x,load_variables):
        #layer 1 - conv
        w_1 = tf.get_variable(shape=[7,7,3,64], name='InceptionV1/Conv2d_1a_7x7/weights')
        h_conv1 = tf.nn.conv2d(x, w_1, strides=[1, 2, 2, 1], padding='SAME')
        #layer 1 - max pool
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1],\
                                 strides=[1, 2, 2, 1], padding='SAME')
        #layer 1 -  BN
        beta_1 = tf.get_variable(shape=[64],name='InceptionV1/Conv2d_1a_7x7/BatchNorm/beta')
        moving_mean_1 = tf.get_variable(shape=[64],name='InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_mean')
        moving_variance_1 = tf.get_variable(shape=[64],name='InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_variance')
        h_bn1 = tf.nn.batch_normalization(h_pool1,offset=beta_1,\
                                              mean=moving_mean_1, \
                                              variance=moving_variance_1,scale=None,variance_epsilon=0.001)
        #layer 2 - conv
        w_2 = tf.get_variable(shape=[1,1,64,64], name='InceptionV1/Conv2d_2b_1x1/weights')
        b_2 = tf.get_variable(shape=[64],name='InceptionV1/Conv2d_2b_1x1/bias')
        h_conv2 = tf.nn.conv2d(h_bn1, w_2, strides=[1, 1, 1, 1], padding='SAME') + b_2
        #layer 2 -  BN
        beta_2 = tf.get_variable(shape=[64],name='InceptionV1/Conv2d_2b_1x1/BatchNorm/beta')
        moving_mean_2 = tf.get_variable(shape=[64],name='InceptionV1/Conv2d_2b_1x1/BatchNorm/moving_mean')
        moving_variance_2 = tf.get_variable(shape=[64],name='InceptionV1/Conv2d_2b_1x1/BatchNorm/moving_variance')
        h_bn2 = tf.nn.batch_normalization(h_conv2,offset=beta_2,\
                                              mean=moving_mean_2, \
                                              variance=moving_variance_2,scale=None,variance_epsilon=0.001)
        #layer 3 - conv
        w_3 = tf.get_variable(shape=[3,3,64,192], name='InceptionV1/Conv2d_2c_3x3/weights')
        b_3 = tf.get_variable(shape=[192],name='InceptionV1/Conv2d_2c_3x3/bias')
        h_conv3 = tf.nn.conv2d(h_bn2, w_3, strides=[1, 1, 1, 1], padding='SAME') + b_3
        #layer 3 - BN
        beta_3 = tf.get_variable(shape=[192],name='InceptionV1/Conv2d_2c_3x3/BatchNorm/beta')
        moving_mean_3 = tf.get_variable(shape=[192],name='InceptionV1/Conv2d_2c_3x3/BatchNorm/moving_mean')
        moving_variance_3 = tf.get_variable(shape=[192],name='InceptionV1/Conv2d_2c_3x3/BatchNorm/moving_variance')
        h_bn3 = tf.nn.batch_normalization(h_conv3,offset=beta_3,\
                                              mean=moving_mean_3, \
                                              variance=moving_variance_3,scale=None,variance_epsilon=0.001)
        #layer 3 - max pool
        h_pool3 = tf.nn.max_pool(h_bn3, ksize=[1, 3, 3, 1], \
                                 strides=[1, 2, 2, 1], padding='SAME')
        #mixed layer 3b
        #first inception
        #branch 0
        w_4 = tf.get_variable(shape=[1,1,192,64], name='InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/weights')
        b_4 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/bias')
        branch1_0 = tf.nn.conv2d(h_pool3, w_4, strides=[1, 1, 1, 1], padding='SAME') + b_4
        beta_4 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_4 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_4 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch1_0 = tf.nn.batch_normalization(branch1_0,offset=beta_4,\
                                              mean=moving_mean_4, \
                                              variance=moving_variance_4,scale=None,variance_epsilon=0.001)
        #branch 1
        w_5 = tf.get_variable(shape=[1,1,192,96], name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/weights')
        b_5 = tf.get_variable(shape=[96],name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/bias')
        branch1_1 = tf.nn.conv2d(h_pool3, w_5, strides=[1, 1, 1, 1], padding='SAME') + b_5
        beta_5 = tf.get_variable(shape=[96],name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_5 = tf.get_variable(shape=[96],name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_5 = tf.get_variable(shape=[96],name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch1_1 = tf.nn.batch_normalization(branch1_1,offset=beta_5,\
                                              mean=moving_mean_5, \
                                              variance=moving_variance_5,scale=None,variance_epsilon=0.001)
        w_6 = tf.get_variable(shape=[3,3,96,128], name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/weights')
        b_6 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/bias')
        branch1_1 = tf.nn.conv2d(branch1_1, w_6, strides=[1, 1, 1, 1], padding='SAME') + b_6
        beta_6 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta')
        moving_mean_6 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean')
        moving_variance_6 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance')
        branch1_1 = tf.nn.batch_normalization(branch1_1,offset=beta_6,\
                                              mean=moving_mean_6, \
                                              variance=moving_variance_6,scale=None,variance_epsilon=0.001)
        #branch 2
        w_7 = tf.get_variable(shape=[1,1,192,16], name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/weights')
        b_7 = tf.get_variable(shape=[16],name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/bias')
        branch1_2 = tf.nn.conv2d(h_pool3, w_7, strides=[1, 1, 1, 1], padding='SAME') + b_7
        beta_7 = tf.get_variable(shape=[16],name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_7 = tf.get_variable(shape=[16],name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_7 = tf.get_variable(shape=[16],name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch1_2 = tf.nn.batch_normalization(branch1_2,offset=beta_7,\
                                              mean=moving_mean_7, \
                                              variance=moving_variance_7,scale=None,variance_epsilon=0.001)
        w_8 = tf.get_variable(shape=[3,3,16,32], name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/weights')
        b_8 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/bias')
        branch1_2 = tf.nn.conv2d(branch1_2, w_8, strides=[1, 1, 1, 1], padding='SAME') + b_8
        beta_8 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/beta')
        moving_mean_8 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean')
        moving_variance_8 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance')
        branch1_2 = tf.nn.batch_normalization(branch1_2,offset=beta_8,\
                                              mean=moving_mean_8, \
                                              variance=moving_variance_8,scale=None,variance_epsilon=0.001)
        #branch 3
        branch1_3 = tf.nn.max_pool(h_pool3, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_9 = tf.get_variable(shape=[1,1,192,32], name='InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/weights')
        b_9 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/bias')
        branch1_3 = tf.nn.conv2d(branch1_3, w_9, strides=[1, 1, 1, 1], padding='SAME') + b_9
        beta_9 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta')
        moving_mean_9 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean')
        moving_variance_9 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance')
        branch1_3 = tf.nn.batch_normalization(branch1_3,offset=beta_9,\
                                              mean=moving_mean_9, \
                                              variance=moving_variance_9,scale=None,variance_epsilon=0.001)
        incpt = tf.concat(
            axis=3, values=[branch1_0, branch1_1, branch1_2, branch1_3])
        #second inception
        #branch 0
        w_10 = tf.get_variable(shape=[1,1,256,128], name='InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/weights')
        b_10 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/bias')
        branch2_0 = tf.nn.conv2d(incpt, w_10, strides=[1, 1, 1, 1], padding='SAME') + b_10
        beta_10 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_10 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_10 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch2_0 = tf.nn.batch_normalization(branch2_0,offset=beta_10,\
                                              mean=moving_mean_10, \
                                              variance=moving_variance_10,scale=None,variance_epsilon=0.001)
        #branch 1
        w_11 = tf.get_variable(shape=[1,1,256,128], name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/weights')
        b_11 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/bias')
        branch2_1 = tf.nn.conv2d(incpt, w_11, strides=[1, 1, 1, 1], padding='SAME') + b_11
        beta_11 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_11 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_11 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch2_1 = tf.nn.batch_normalization(branch2_1,offset=beta_11,\
                                              mean=moving_mean_11, \
                                              variance=moving_variance_11,scale=None,variance_epsilon=0.001)
        w_12 = tf.get_variable(shape=[3,3,128,192], name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/weights')
        b_12 = tf.get_variable(shape=[192],name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/bias')
        branch2_1 = tf.nn.conv2d(branch2_1, w_12, strides=[1, 1, 1, 1], padding='SAME') + b_12
        beta_12 = tf.get_variable(shape=[192],name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta')
        moving_mean_12 = tf.get_variable(shape=[192],name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean')
        moving_variance_12 = tf.get_variable(shape=[192],name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance')
        branch2_1 = tf.nn.batch_normalization(branch2_1,offset=beta_12,\
                                              mean=moving_mean_12, \
                                              variance=moving_variance_12,scale=None,variance_epsilon=0.001)
        #branch 2
        w_13 = tf.get_variable(shape=[1,1,256,32], name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/weights')
        b_13 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/bias')
        branch2_2 = tf.nn.conv2d(incpt, w_13, strides=[1, 1, 1, 1], padding='SAME') + b_13
        beta_13 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_13 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_13 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch2_2 = tf.nn.batch_normalization(branch2_2,offset=beta_13,\
                                              mean=moving_mean_13, \
                                              variance=moving_variance_13,scale=None,variance_epsilon=0.001)
        w_14 = tf.get_variable(shape=[3,3,32,96], name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/weights')
        b_14 = tf.get_variable(shape=[96],name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/bias')
        branch2_2 = tf.nn.conv2d(branch2_2, w_14, strides=[1, 1, 1, 1], padding='SAME') + b_14
        beta_14 = tf.get_variable(shape=[96],name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta')
        moving_mean_14 = tf.get_variable(shape=[96],name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean')
        moving_variance_14 = tf.get_variable(shape=[96],name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance')
        branch2_2 = tf.nn.batch_normalization(branch2_2,offset=beta_14,\
                                              mean=moving_mean_14, \
                                              variance=moving_variance_14,scale=None,variance_epsilon=0.001)
        #branch 3
        branch2_3 = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_15 = tf.get_variable(shape=[1,1,256,64], name='InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/weights')
        b_15 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/bias')
        branch2_3 = tf.nn.conv2d(branch2_3, w_15, strides=[1, 1, 1, 1], padding='SAME') + b_15
        beta_15 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta')
        moving_mean_15 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean')
        moving_variance_15 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance')
        branch2_3 = tf.nn.batch_normalization(branch2_3,offset=beta_15,\
                                              mean=moving_mean_15, \
                                              variance=moving_variance_15,scale=None,variance_epsilon=0.001)
        incpt = tf.concat(
            axis=3, values=[branch2_0, branch2_1, branch2_2, branch2_3])
        incpt = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                   strides=[1, 2, 2, 1], padding='SAME')
        #third inception
        #branch 0
        w_16 = tf.get_variable(shape=[1,1,480,192], name='InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/weights')
        b_16 = tf.get_variable(shape=[192],name='InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/bias')
        branch3_0 = tf.nn.conv2d(incpt, w_16, strides=[1, 1, 1, 1], padding='SAME') + b_16
        beta_16 = tf.get_variable(shape=[192],name='InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_16 = tf.get_variable(shape=[192],name='InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_16 = tf.get_variable(shape=[192],name='InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch3_0 = tf.nn.batch_normalization(branch3_0,offset=beta_16,\
                                              mean=moving_mean_16, \
                                              variance=moving_variance_16,scale=None,variance_epsilon=0.001)
        #branch 1
        w_17 = tf.get_variable(shape=[1,1,480,96], name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/weights')
        b_17 = tf.get_variable(shape=[96],name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/bias')
        branch3_1 = tf.nn.conv2d(incpt, w_17, strides=[1, 1, 1, 1], padding='SAME') + b_17
        beta_17 = tf.get_variable(shape=[96],name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_17 = tf.get_variable(shape=[96],name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_17 = tf.get_variable(shape=[96],name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch3_1 = tf.nn.batch_normalization(branch3_1,offset=beta_17,\
                                              mean=moving_mean_17, \
                                              variance=moving_variance_17,scale=None,variance_epsilon=0.001)
        w_18 = tf.get_variable(shape=[3,3,96,208], name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/weights')
        b_18 = tf.get_variable(shape=[208],name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/bias')
        branch3_1 = tf.nn.conv2d(branch3_1, w_18, strides=[1, 1, 1, 1], padding='SAME') + b_18
        beta_18 = tf.get_variable(shape=[208],name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta')
        moving_mean_18 = tf.get_variable(shape=[208],name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean')
        moving_variance_18 = tf.get_variable(shape=[208],name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance')
        branch3_1 = tf.nn.batch_normalization(branch3_1,offset=beta_18,\
                                              mean=moving_mean_18, \
                                              variance=moving_variance_18,scale=None,variance_epsilon=0.001)
        #branch 2
        w_19 = tf.get_variable(shape=[1,1,480,16], name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/weights')
        b_19 = tf.get_variable(shape=[16],name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/bias')
        branch3_2 = tf.nn.conv2d(incpt, w_19, strides=[1, 1, 1, 1], padding='SAME') + b_19
        beta_19 = tf.get_variable(shape=[16],name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_19 = tf.get_variable(shape=[16],name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_19 = tf.get_variable(shape=[16],name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch3_2 = tf.nn.batch_normalization(branch3_2,offset=beta_19,\
                                              mean=moving_mean_19, \
                                              variance=moving_variance_19,scale=None,variance_epsilon=0.001)
        w_20 = tf.get_variable(shape=[3,3,16,48], name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/weights')
        b_20 = tf.get_variable(shape=[48],name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/bias')
        branch3_2 = tf.nn.conv2d(branch3_2, w_20, strides=[1, 1, 1, 1], padding='SAME') + b_20
        beta_20 = tf.get_variable(shape=[48],name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/beta')
        moving_mean_20 = tf.get_variable(shape=[48],name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean')
        moving_variance_20 = tf.get_variable(shape=[48],name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance')
        branch3_2 = tf.nn.batch_normalization(branch3_2,offset=beta_20,\
                                              mean=moving_mean_20, \
                                              variance=moving_variance_20,scale=None,variance_epsilon=0.001)
        #branch 3
        branch3_3 = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_21 = tf.get_variable(shape=[1,1,480,64], name='InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/weights')
        b_21 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/bias')
        branch3_3 = tf.nn.conv2d(branch3_3, w_21, strides=[1, 1, 1, 1], padding='SAME') + b_21
        beta_21 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta')
        moving_mean_21 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean')
        moving_variance_21 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance')
        branch3_3 = tf.nn.batch_normalization(branch3_3,offset=beta_21,\
                                              mean=moving_mean_21, \
                                              variance=moving_variance_21,scale=None,variance_epsilon=0.001)
        incpt = tf.concat(
            axis=3, values=[branch3_0, branch3_1, branch3_2, branch3_3])
        #fourth inception
        #branch 0
        w_22 = tf.get_variable(shape=[1,1,512,160], name='InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/weights')
        b_22 = tf.get_variable(shape=[160],name='InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/bias')
        branch4_0 = tf.nn.conv2d(incpt, w_22, strides=[1, 1, 1, 1], padding='SAME') + b_22
        beta_22 = tf.get_variable(shape=[160],name='InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_22 = tf.get_variable(shape=[160],name='InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_22 = tf.get_variable(shape=[160],name='InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch4_0 = tf.nn.batch_normalization(branch4_0,offset=beta_22,\
                                              mean=moving_mean_22, \
                                              variance=moving_variance_22,scale=None,variance_epsilon=0.001)
        #branch 1
        w_23 = tf.get_variable(shape=[1,1,512,112], name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/weights')
        b_23 = tf.get_variable(shape=[112],name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/bias')
        branch4_1 = tf.nn.conv2d(incpt, w_23, strides=[1, 1, 1, 1], padding='SAME') + b_23
        beta_23 = tf.get_variable(shape=[112],name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_23 = tf.get_variable(shape=[112],name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_23 = tf.get_variable(shape=[112],name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch4_1 = tf.nn.batch_normalization(branch4_1,offset=beta_23,\
                                              mean=moving_mean_23, \
                                              variance=moving_variance_23,scale=None,variance_epsilon=0.001)
        w_24 = tf.get_variable(shape=[3,3,112,224], name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/weights')
        b_24 = tf.get_variable(shape=[224],name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/bias')
        branch4_1 = tf.nn.conv2d(branch4_1, w_24, strides=[1, 1, 1, 1], padding='SAME') + b_24
        beta_24 = tf.get_variable(shape=[224],name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta')
        moving_mean_24 = tf.get_variable(shape=[224],name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean')
        moving_variance_24 = tf.get_variable(shape=[224],name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance')
        branch4_1 = tf.nn.batch_normalization(branch4_1,offset=beta_24,\
                                              mean=moving_mean_24, \
                                              variance=moving_variance_24,scale=None,variance_epsilon=0.001)
        #branch 2
        w_25 = tf.get_variable(shape=[1,1,512,24], name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/weights')
        b_25 = tf.get_variable(shape=[24],name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/bias')
        branch4_2 = tf.nn.conv2d(incpt, w_25, strides=[1, 1, 1, 1], padding='SAME') + b_25
        beta_25 = tf.get_variable(shape=[24],name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_25 = tf.get_variable(shape=[24],name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_25 = tf.get_variable(shape=[24],name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch4_2 = tf.nn.batch_normalization(branch4_2,offset=beta_25,\
                                              mean=moving_mean_25, \
                                              variance=moving_variance_25,scale=None,variance_epsilon=0.001)
        w_26 = tf.get_variable(shape=[3,3,24,64], name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/weights')
        b_26 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/bias')
        branch4_2 = tf.nn.conv2d(branch4_2, w_26, strides=[1, 1, 1, 1], padding='SAME') + b_26
        beta_26 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta')
        moving_mean_26 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean')
        moving_variance_26 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance')
        branch4_2 = tf.nn.batch_normalization(branch4_2,offset=beta_26,\
                                              mean=moving_mean_26, \
                                              variance=moving_variance_26,scale=None,variance_epsilon=0.001)
        #branch 3
        branch4_3 = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_27 = tf.get_variable(shape=[1,1,512,64], name='InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/weights')
        b_27 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/bias')
        branch4_3 = tf.nn.conv2d(branch4_3, w_27, strides=[1, 1, 1, 1], padding='SAME') + b_27
        beta_27 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta')
        moving_mean_27 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean')
        moving_variance_27 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance')
        branch4_3 = tf.nn.batch_normalization(branch4_3,offset=beta_27,\
                                              mean=moving_mean_27, \
                                              variance=moving_variance_27,scale=None,variance_epsilon=0.001)
        incpt = tf.concat(
            axis=3, values=[branch4_0, branch4_1, branch4_2, branch4_3])
        #fifth inception
        #branch 0
        w_28 = tf.get_variable(shape=[1,1,512,128], name='InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/weights')
        b_28 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/bias')
        branch5_0 = tf.nn.conv2d(incpt, w_28, strides=[1, 1, 1, 1], padding='SAME') + b_28
        beta_28 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_28 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_28 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch5_0 = tf.nn.batch_normalization(branch5_0,offset=beta_28,\
                                              mean=moving_mean_28, \
                                              variance=moving_variance_28,scale=None,variance_epsilon=0.001)
        #branch 1
        w_29 = tf.get_variable(shape=[1,1,512,128], name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/weights')
        b_29 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/bias')
        branch5_1 = tf.nn.conv2d(incpt, w_29, strides=[1, 1, 1, 1], padding='SAME') + b_29
        beta_29 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_29 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_29 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch5_1 = tf.nn.batch_normalization(branch5_1,offset=beta_29,\
                                              mean=moving_mean_29, \
                                              variance=moving_variance_29,scale=None,variance_epsilon=0.001)
        w_30 = tf.get_variable(shape=[3,3,128,256], name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/weights')
        b_30 = tf.get_variable(shape=[256],name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/bias')
        branch5_1 = tf.nn.conv2d(branch5_1, w_30, strides=[1, 1, 1, 1], padding='SAME') + b_30
        beta_30 = tf.get_variable(shape=[256],name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/beta')
        moving_mean_30 = tf.get_variable(shape=[256],name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean')
        moving_variance_30 = tf.get_variable(shape=[256],name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance')
        branch5_1 = tf.nn.batch_normalization(branch5_1,offset=beta_30,\
                                              mean=moving_mean_30, \
                                              variance=moving_variance_30,scale=None,variance_epsilon=0.001)
        #branch 2
        w_31 = tf.get_variable(shape=[1,1,512,24], name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/weights')
        b_31 = tf.get_variable(shape=[24],name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/bias')
        branch5_2 = tf.nn.conv2d(incpt, w_31, strides=[1, 1, 1, 1], padding='SAME') + b_31
        beta_31 = tf.get_variable(shape=[24],name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_31 = tf.get_variable(shape=[24],name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_31 = tf.get_variable(shape=[24],name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch5_2 = tf.nn.batch_normalization(branch5_2,offset=beta_31,\
                                              mean=moving_mean_31, \
                                              variance=moving_variance_31,scale=None,variance_epsilon=0.001)
        w_32 = tf.get_variable(shape=[3,3,24,64], name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/weights')
        b_32 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/bias')
        branch5_2 = tf.nn.conv2d(branch5_2, w_32, strides=[1, 1, 1, 1], padding='SAME') + b_32
        beta_32 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/beta')
        moving_mean_32 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean')
        moving_variance_32 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance')
        branch5_2 = tf.nn.batch_normalization(branch5_2,offset=beta_32,\
                                              mean=moving_mean_32, \
                                              variance=moving_variance_32,scale=None,variance_epsilon=0.001)
        #branch 3
        branch5_3 = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_33 = tf.get_variable(shape=[1,1,512,64], name='InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/weights')
        b_33 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/bias')
        branch5_3 = tf.nn.conv2d(branch5_3, w_33, strides=[1, 1, 1, 1], padding='SAME') + b_33
        beta_33 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/beta')
        moving_mean_33 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean')
        moving_variance_33 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance')
        branch5_3 = tf.nn.batch_normalization(branch5_3,offset=beta_33,\
                                              mean=moving_mean_33, \
                                              variance=moving_variance_33,scale=None,variance_epsilon=0.001)
        incpt = tf.concat(
            axis=3, values=[branch5_0, branch5_1, branch5_2, branch5_3])
        #sixth inception
        #branch 0
        w_34 = tf.get_variable(shape=[1,1,512,112], name='InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/weights')
        b_34 = tf.get_variable(shape=[112],name='InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/bias')
        branch6_0 = tf.nn.conv2d(incpt, w_34, strides=[1, 1, 1, 1], padding='SAME') + b_34
        beta_34 = tf.get_variable(shape=[112],name='InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_34 = tf.get_variable(shape=[112],name='InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_34 = tf.get_variable(shape=[112],name='InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch6_0 = tf.nn.batch_normalization(branch6_0,offset=beta_34,\
                                              mean=moving_mean_34, \
                                              variance=moving_variance_34,scale=None,variance_epsilon=0.001)
        #branch 1
        w_35 = tf.get_variable(shape=[1,1,512,144], name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/weights')
        b_35 = tf.get_variable(shape=[144],name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/bias')
        branch6_1 = tf.nn.conv2d(incpt, w_35, strides=[1, 1, 1, 1], padding='SAME') + b_35
        beta_35 = tf.get_variable(shape=[144],name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_35 = tf.get_variable(shape=[144],name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_35 = tf.get_variable(shape=[144],name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch6_1 = tf.nn.batch_normalization(branch6_1,offset=beta_35,\
                                              mean=moving_mean_35, \
                                              variance=moving_variance_35,scale=None,variance_epsilon=0.001)
        w_36 = tf.get_variable(shape=[3,3,144,288], name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/weights')
        b_36 = tf.get_variable(shape=[288],name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/bias')
        branch6_1 = tf.nn.conv2d(branch6_1, w_36, strides=[1, 1, 1, 1], padding='SAME') + b_36
        beta_36 = tf.get_variable(shape=[288],name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/beta')
        moving_mean_36 = tf.get_variable(shape=[288],name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean')
        moving_variance_36 = tf.get_variable(shape=[288],name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance')
        branch6_1 = tf.nn.batch_normalization(branch6_1,offset=beta_36,\
                                              mean=moving_mean_36, \
                                              variance=moving_variance_36,scale=None,variance_epsilon=0.001)
        #branch 2
        w_37 = tf.get_variable(shape=[1,1,512,32], name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/weights')
        b_37 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/bias')
        branch6_2 = tf.nn.conv2d(incpt, w_37, strides=[1, 1, 1, 1], padding='SAME') + b_37
        beta_37 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_37 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_37 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch6_2 = tf.nn.batch_normalization(branch6_2,offset=beta_37,\
                                              mean=moving_mean_37, \
                                              variance=moving_variance_37,scale=None,variance_epsilon=0.001)
        w_38 = tf.get_variable(shape=[3,3,32,64], name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/weights')
        b_38 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/bias')
        branch6_2 = tf.nn.conv2d(branch6_2, w_38, strides=[1, 1, 1, 1], padding='SAME') + b_38
        beta_38 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/beta')
        moving_mean_38 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean')
        moving_variance_38 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance')
        branch6_2 = tf.nn.batch_normalization(branch6_2,offset=beta_38,\
                                              mean=moving_mean_38, \
                                              variance=moving_variance_38,scale=None,variance_epsilon=0.001)
        #branch 3
        branch6_3 = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_39 = tf.get_variable(shape=[1,1,512,64], name='InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/weights')
        b_39 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/bias')
        branch6_3 = tf.nn.conv2d(branch6_3, w_39, strides=[1, 1, 1, 1], padding='SAME') + b_39
        beta_39 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/beta')
        moving_mean_39 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean')
        moving_variance_39 = tf.get_variable(shape=[64],name='InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance')
        branch6_3 = tf.nn.batch_normalization(branch6_3,offset=beta_39,\
                                              mean=moving_mean_39, \
                                              variance=moving_variance_39,scale=None,variance_epsilon=0.001)
        incpt = tf.concat(
            axis=3, values=[branch6_0, branch6_1, branch6_2, branch6_3])
        #seventh inception
        #branch 0
        w_40 = tf.get_variable(shape=[1,1,528,256], name='InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/weights')
        b_40 = tf.get_variable(shape=[256],name='InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/bias')
        branch7_0 = tf.nn.conv2d(incpt, w_40, strides=[1, 1, 1, 1], padding='SAME') + b_40
        beta_40 = tf.get_variable(shape=[256],name='InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_40 = tf.get_variable(shape=[256],name='InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_40 = tf.get_variable(shape=[256],name='InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch7_0 = tf.nn.batch_normalization(branch7_0,offset=beta_40,\
                                              mean=moving_mean_40, \
                                              variance=moving_variance_40,scale=None,variance_epsilon=0.001)
        #branch 1
        w_41 = tf.get_variable(shape=[1,1,528,160], name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/weights')
        b_41 = tf.get_variable(shape=[160],name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/bias')
        branch7_1 = tf.nn.conv2d(incpt, w_41, strides=[1, 1, 1, 1], padding='SAME') + b_41
        beta_41 = tf.get_variable(shape=[160],name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_41 = tf.get_variable(shape=[160],name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_41 = tf.get_variable(shape=[160],name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch7_1 = tf.nn.batch_normalization(branch7_1,offset=beta_41,\
                                              mean=moving_mean_41, \
                                              variance=moving_variance_41,scale=None,variance_epsilon=0.001)
        w_42 = tf.get_variable(shape=[3,3,160,320], name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/weights')
        b_42 = tf.get_variable(shape=[320],name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/bias')
        branch7_1 = tf.nn.conv2d(branch7_1, w_42, strides=[1, 1, 1, 1], padding='SAME') + b_42
        beta_42 = tf.get_variable(shape=[320],name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/beta')
        moving_mean_42 = tf.get_variable(shape=[320],name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean')
        moving_variance_42 = tf.get_variable(shape=[320],name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance')
        branch7_1 = tf.nn.batch_normalization(branch7_1,offset=beta_42,\
                                              mean=moving_mean_42, \
                                              variance=moving_variance_42,scale=None,variance_epsilon=0.001)
        #branch 2
        w_43 = tf.get_variable(shape=[1,1,528,32], name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/weights')
        b_43 = tf.get_variable(shape=[32],name='IInceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/bias')
        branch7_2 = tf.nn.conv2d(incpt, w_43, strides=[1, 1, 1, 1], padding='SAME') + b_43
        beta_43 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_43 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_43 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch7_2 = tf.nn.batch_normalization(branch7_2,offset=beta_43,\
                                              mean=moving_mean_43, \
                                              variance=moving_variance_43,scale=None,variance_epsilon=0.001)
        w_44 = tf.get_variable(shape=[3,3,32,128], name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/weights')
        b_44 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/bias')
        branch7_2 = tf.nn.conv2d(branch7_2, w_44, strides=[1, 1, 1, 1], padding='SAME') + b_44
        beta_44 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/beta')
        moving_mean_44 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean')
        moving_variance_44 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance')
        branch7_2 = tf.nn.batch_normalization(branch7_2,offset=beta_44,\
                                              mean=moving_mean_44, \
                                              variance=moving_variance_44,scale=None,variance_epsilon=0.001)
        #branch 3
        branch7_3 = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_45 = tf.get_variable(shape=[1,1,528,128], name='InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/weights')
        b_45 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/bias')
        branch7_3 = tf.nn.conv2d(branch7_3, w_45, strides=[1, 1, 1, 1], padding='SAME') + b_45
        beta_45 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/beta')
        moving_mean_45 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean')
        moving_variance_45 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance')
        branch7_3 = tf.nn.batch_normalization(branch7_3,offset=beta_45,\
                                              mean=moving_mean_45, \
                                              variance=moving_variance_45,scale=None,variance_epsilon=0.001)
        incpt = tf.concat(
            axis=3, values=[branch7_0, branch7_1, branch7_2, branch7_3])
        incpt = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                               strides=[1, 2, 2, 1], padding='SAME')
        #eighth inception
        #branch 0
        w_46 = tf.get_variable(shape=[1,1,832,256], name='InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/weights')
        b_46 = tf.get_variable(shape=[256],name='InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/bias')
        branch8_0 = tf.nn.conv2d(incpt, w_46, strides=[1, 1, 1, 1], padding='SAME') + b_46
        beta_46 = tf.get_variable(shape=[256],name='InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_46 = tf.get_variable(shape=[256],name='InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_46 = tf.get_variable(shape=[256],name='InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch8_0 = tf.nn.batch_normalization(branch8_0,offset=beta_46,\
                                              mean=moving_mean_46, \
                                              variance=moving_variance_46,scale=None,variance_epsilon=0.001)
        #branch 1
        w_47 = tf.get_variable(shape=[1,1,832,160], name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/weights')
        b_47 = tf.get_variable(shape=[160],name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/bias')
        branch8_1 = tf.nn.conv2d(incpt, w_47, strides=[1, 1, 1, 1], padding='SAME') + b_47
        beta_47 = tf.get_variable(shape=[160],name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_47 = tf.get_variable(shape=[160],name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_47 = tf.get_variable(shape=[160],name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch8_1 = tf.nn.batch_normalization(branch8_1,offset=beta_47,\
                                              mean=moving_mean_47, \
                                              variance=moving_variance_47,scale=None,variance_epsilon=0.001)
        w_48 = tf.get_variable(shape=[3,3,160,320], name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/weights')
        b_48 = tf.get_variable(shape=[320],name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/bias')
        branch8_1 = tf.nn.conv2d(branch8_1, w_48, strides=[1, 1, 1, 1], padding='SAME') + b_48
        beta_48 = tf.get_variable(shape=[320],name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta')
        moving_mean_48 = tf.get_variable(shape=[320],name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean')
        moving_variance_48 = tf.get_variable(shape=[320],name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance')
        branch8_1 = tf.nn.batch_normalization(branch8_1,offset=beta_48,\
                                              mean=moving_mean_48, \
                                              variance=moving_variance_48,scale=None,variance_epsilon=0.001)
        #branch 2
        w_49 = tf.get_variable(shape=[1,1,832,32], name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/weights')
        b_49 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/bias')
        branch8_2 = tf.nn.conv2d(incpt, w_49, strides=[1, 1, 1, 1], padding='SAME') + b_49
        beta_49 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_49 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_49 = tf.get_variable(shape=[32],name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch8_2 = tf.nn.batch_normalization(branch8_2,offset=beta_49,\
                                              mean=moving_mean_49, \
                                              variance=moving_variance_49,scale=None,variance_epsilon=0.001)
        w_50 = tf.get_variable(shape=[3,3,32,128], name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/weights')
        b_50 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/bias')
        branch8_2 = tf.nn.conv2d(branch8_2, w_50, strides=[1, 1, 1, 1], padding='SAME') + b_50
        beta_50 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/beta')
        moving_mean_50 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/moving_mean')
        moving_variance_50 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/moving_variance')
        branch8_2 = tf.nn.batch_normalization(branch8_2,offset=beta_50,\
                                              mean=moving_mean_50, \
                                              variance=moving_variance_50,scale=None,variance_epsilon=0.001)
        #branch 3
        branch8_3 = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_51 = tf.get_variable(shape=[1,1,832,128], name='InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/weights')
        b_51 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/bias')
        branch8_3 = tf.nn.conv2d(branch8_3, w_51, strides=[1, 1, 1, 1], padding='SAME') + b_51
        beta_51 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta')
        moving_mean_51 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean')
        moving_variance_51 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance')
        branch8_3 = tf.nn.batch_normalization(branch8_3,offset=beta_51,\
                                              mean=moving_mean_51, \
                                              variance=moving_variance_51,scale=None,variance_epsilon=0.001)
        incpt = tf.concat(
            axis=3, values=[branch8_0, branch8_1, branch8_2, branch8_3])
        #ninth inception
        #branch 0
        w_52 = tf.get_variable(shape=[1,1,832,384], name='InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/weights')
        b_52 = tf.get_variable(shape=[384],name='InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/bias')
        branch9_0 = tf.nn.conv2d(incpt, w_52, strides=[1, 1, 1, 1], padding='SAME') + b_52
        beta_52 = tf.get_variable(shape=[384],name='InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_52 = tf.get_variable(shape=[384],name='InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_52 = tf.get_variable(shape=[384],name='InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch9_0 = tf.nn.batch_normalization(branch9_0,offset=beta_52,\
                                              mean=moving_mean_52, \
                                              variance=moving_variance_52,scale=None,variance_epsilon=0.001)
        #branch 1
        w_53 = tf.get_variable(shape=[1,1,832,192], name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/weights')
        b_53 = tf.get_variable(shape=[192],name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/bias')
        branch9_1 = tf.nn.conv2d(incpt, w_53, strides=[1, 1, 1, 1], padding='SAME') + b_53
        beta_53 = tf.get_variable(shape=[192],name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_53 = tf.get_variable(shape=[192],name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_53 = tf.get_variable(shape=[192],name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch9_1 = tf.nn.batch_normalization(branch9_1,offset=beta_53,\
                                              mean=moving_mean_53, \
                                              variance=moving_variance_53,scale=None,variance_epsilon=0.001)
        w_54 = tf.get_variable(shape=[3,3,192,384], name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/weights')
        b_54 = tf.get_variable(shape=[384],name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/bias')
        branch9_1 = tf.nn.conv2d(branch9_1, w_54, strides=[1, 1, 1, 1], padding='SAME') + b_54
        beta_54 = tf.get_variable(shape=[384],name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta')
        moving_mean_54 = tf.get_variable(shape=[384],name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean')
        moving_variance_54 = tf.get_variable(shape=[384],name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance')
        branch9_1 = tf.nn.batch_normalization(branch9_1,offset=beta_54,\
                                              mean=moving_mean_54, \
                                              variance=moving_variance_54,scale=None,variance_epsilon=0.001)
        #branch 2
        w_55 = tf.get_variable(shape=[1,1,832,48], name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/weights')
        b_55 = tf.get_variable(shape=[48],name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/bias')
        branch9_2 = tf.nn.conv2d(incpt, w_55, strides=[1, 1, 1, 1], padding='SAME') + b_55
        beta_55 = tf.get_variable(shape=[48],name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta')
        moving_mean_55 = tf.get_variable(shape=[48],name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean')
        moving_variance_55 = tf.get_variable(shape=[48],name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance')
        branch9_2 = tf.nn.batch_normalization(branch9_2,offset=beta_55,\
                                              mean=moving_mean_55, \
                                              variance=moving_variance_55,scale=None,variance_epsilon=0.001)
        w_56 = tf.get_variable(shape=[3,3,48,128], name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/weights')
        b_56 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/bias')
        branch9_2 = tf.nn.conv2d(branch9_2, w_56, strides=[1, 1, 1, 1], padding='SAME') + b_56
        beta_56 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta')
        moving_mean_56 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean')
        moving_variance_56 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance')
        branch9_2 = tf.nn.batch_normalization(branch9_2,offset=beta_56,\
                                              mean=moving_mean_56, \
                                              variance=moving_variance_56,scale=None,variance_epsilon=0.001)
        #branch 3
        branch9_3 = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_57 = tf.get_variable(shape=[1,1,832,128], name='InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/weights')
        b_57 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/bias')
        branch9_3 = tf.nn.conv2d(branch9_3, w_57, strides=[1, 1, 1, 1], padding='SAME') + b_57
        beta_57 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta')
        moving_mean_57 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean')
        moving_variance_57 = tf.get_variable(shape=[128],name='InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance')
        branch9_3 = tf.nn.batch_normalization(branch9_3,offset=beta_57,\
                                              mean=moving_mean_57, \
                                              variance=moving_variance_57,scale=None,variance_epsilon=0.001)
        nets = tf.concat(
            axis=3, values=[branch9_0, branch9_1, branch9_2, branch9_3])
        nets = tf.nn.avg_pool(nets, ksize=[1, 7, 7, 1], \
                       strides=[1, 1, 1, 1], padding='VALID')

        #fc layer
        w_58 = tf.get_variable(shape=[1024,64],name='fc_layer_0/weights')
        b_58 = tf.get_variable(shape=[64],name='fc_layer_0/bias')
        nets = tf.reshape(nets,[-1,1024])
        nets = tf.add(tf.matmul(nets,w_58),b_58)

        if load_variables:
            saver = tf.train.Saver([w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10,
                                    w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20,
                                    w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30,
                                    w_31, w_32, w_33, w_34, w_35, w_36, w_37, w_38, w_39, w_40,
                                    w_41, w_42, w_43, w_44, w_45, w_46, w_47, w_48, w_49, w_50
                                       , w_51, w_52, w_53, w_54, w_55, w_56, w_57,
                                    beta_1, beta_2, beta_3, beta_4, beta_5, beta_6, beta_7, beta_8, beta_9, beta_10,
                                    beta_11, beta_12, beta_13, beta_14, beta_15, beta_16, beta_17, beta_18, beta_19,
                                    beta_20,
                                    beta_21, beta_22, beta_23, beta_24, beta_25, beta_26, beta_27, beta_28, beta_29,
                                    beta_30,
                                    beta_31, beta_32, beta_33, beta_34, beta_35, beta_36, beta_37, beta_38, beta_39,
                                    beta_40,
                                    beta_41, beta_42, beta_43, beta_44, beta_45, beta_46, beta_47, beta_48, beta_49,
                                    beta_50,
                                    beta_51, beta_52, beta_53, beta_54, beta_55, beta_56, beta_57,
                                    moving_mean_1, moving_mean_2, moving_mean_3, moving_mean_4, moving_mean_5,
                                    moving_mean_6, moving_mean_7, moving_mean_8, moving_mean_9, moving_mean_10,
                                    moving_mean_11, moving_mean_12, moving_mean_13, moving_mean_14, moving_mean_15,
                                    moving_mean_16, moving_mean_17, moving_mean_18, moving_mean_19, moving_mean_20,
                                    moving_mean_21, moving_mean_22, moving_mean_23, moving_mean_24, moving_mean_25,
                                    moving_mean_26, moving_mean_27, moving_mean_28, moving_mean_29, moving_mean_30,
                                    moving_mean_31, moving_mean_32, moving_mean_33, moving_mean_34, moving_mean_35,
                                    moving_mean_36, moving_mean_37, moving_mean_38, moving_mean_39, moving_mean_40,
                                    moving_mean_41, moving_mean_42, moving_mean_43, moving_mean_44, moving_mean_45,
                                    moving_mean_46, moving_mean_47, moving_mean_48, moving_mean_49, moving_mean_50,
                                    moving_mean_51, moving_mean_52, moving_mean_53, moving_mean_54, moving_mean_55,
                                    moving_mean_56, moving_mean_57,
                                    moving_variance_1, moving_variance_2, moving_variance_3, moving_variance_4,
                                    moving_variance_5,
                                    moving_variance_6, moving_variance_7, moving_variance_8, moving_variance_9,
                                    moving_variance_10,
                                    moving_variance_11, moving_variance_12, moving_variance_13, moving_variance_14,
                                    moving_variance_15,
                                    moving_variance_16, moving_variance_17, moving_variance_18, moving_variance_19,
                                    moving_variance_20,
                                    moving_variance_21, moving_variance_22, moving_variance_23, moving_variance_24,
                                    moving_variance_25,
                                    moving_variance_26, moving_variance_27, moving_variance_28, moving_variance_29,
                                    moving_variance_30,
                                    moving_variance_31, moving_variance_32, moving_variance_33, moving_variance_34,
                                    moving_variance_35,
                                    moving_variance_36, moving_variance_37, moving_variance_38, moving_variance_39,
                                    moving_variance_40,
                                    moving_variance_41, moving_variance_42, moving_variance_43, moving_variance_44,
                                    moving_variance_45,
                                    moving_variance_46, moving_variance_47, moving_variance_48, moving_variance_49,
                                    moving_variance_50,
                                    moving_variance_51, moving_variance_52, moving_variance_53, moving_variance_54,
                                    moving_variance_55,
                                    moving_variance_56, moving_variance_57])

            init_rem_vars_op = tf.variables_initializer([b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9, b_10,
                                                     b_11, b_12, b_13, b_14, b_15, b_16, b_17, b_18, b_19, b_20,
                                                     b_21, b_22, b_23, b_24, b_25, b_26, b_27, b_28, b_29, b_30,
                                                     b_31, b_32, b_33, b_34, b_35, b_36, b_37, b_38, b_39, b_40,
                                                     b_41, b_42, b_43, b_44, b_45, b_46, b_47, b_48, b_49, b_50,
                                                     b_51, b_52, b_53, b_54, b_55, b_56, b_57, b_58,
                                                     w_58])
            self.sess.run(init_rem_vars_op)
            saver.restore(self.sess, "./inception_v1.ckpt")
        return nets

    def triplet_loss(slef,margins, oa, op, on):
        margin_0 = margins[0]
        margin_1 = margins[1]
        margin_2 = margins[2]

        eucd_p = tf.pow(tf.subtract(oa, op), 2)
        eucd_p = tf.reduce_sum(eucd_p, 1)
        eucd_p = tf.sqrt(eucd_p + 1e-6)

        eucd_n1 = tf.pow(tf.subtract(oa, on), 2)
        eucd_n1 = tf.reduce_sum(eucd_n1, 1)
        eucd_n1 = tf.sqrt(eucd_n1 + 1e-6)

        eucd_n2 = tf.pow(tf.subtract(op, on), 2)
        eucd_n2 = tf.reduce_sum(eucd_n2, 1)
        eucd_n2 = tf.sqrt(eucd_n2 + 1e-6)

        random_negative_margin = tf.constant(margin_0)
        rand_neg = tf.pow(tf.maximum(tf.subtract(random_negative_margin,
                                                 tf.minimum(eucd_n1, eucd_n2)), 0), 2)

        positive_margin = tf.constant(margin_1)

        with tf.name_scope('all_loss'):
            # invertable loss for standard patches
            with tf.name_scope('rand_neg'):
                rand_neg = tf.pow(tf.maximum(tf.subtract(random_negative_margin,
                                                         tf.minimum(eucd_n1, eucd_n2)), 0), 2)
            # covariance loss for transformed patches
            with tf.name_scope('pos'):
                pos = tf.pow(tf.maximum(tf.subtract(positive_margin,
                                                    tf.subtract(tf.minimum(eucd_n1, eucd_n2), eucd_p)), 0), 2)
            # total loss
            with tf.name_scope('loss'):
                losses = rand_neg + pos
                loss = tf.reduce_mean(losses)

        # write summary
        tf.summary.scalar('random_negative_loss', rand_neg)
        tf.summary.scalar('positive_loss', pos)
        tf.summary.scalar('total_loss', loss)

        return loss, eucd_p, eucd_n1, eucd_n2

if __name__ == "__main__":

    sess = tf.Session()
    cnn_triplet = CNN_Triplet_Metric(sess=sess)


