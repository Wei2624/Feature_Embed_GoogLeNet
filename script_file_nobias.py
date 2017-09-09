import h5py
import numpy as np
import tensorflow as tf
import scipy.io
import collections
import random


class CNN_Triplet_Metric(object):
    def __init__(self,sess):
        self.var_dict = self.Variables_Dict()
        img_a = tf.placeholder(tf.float32, [None, 224, 224, 3])
        img_p = tf.placeholder(tf.float32, [None, 224, 224, 3])
        img_n = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.sess = sess



        image_mean = scipy.io.loadmat('image_mean.mat')
        image_mean = image_mean['image_mean']
        image_mean = np.expand_dims(image_mean,axis=0)


        # reading matlab v7.3 file using h5py. it has struct with img as a member
        with h5py.File("training_images_crop15_square256.mat") as f:
            img_data = [f[element[0]][:] for element in f['training_images/img']]
            class_id = [f[element[0]][:] for element in f['training_images/class_id']]

        # img_data = np.float32(np.asarray(img_data))
        img_data = np.asarray(img_data)
        class_id = np.asarray(class_id)
        img_data = np.transpose(img_data, (0, 2, 3, 1))
        class_label = class_id[:, 0, 0]

        # for i in range(len(class_label)-1):
        #     img_data[i,:,:,:] -= np.float32(image_mean)

        index_a, index_p, index_n = self.generate_triplet(class_label, 40)

        # asaved = np.genfromtxt('asaved.csv', delimiter=' ')
        # psaved = np.genfromtxt('psaved.csv', delimiter=' ')
        # nsaved = np.genfromtxt('nsaved.csv', delimiter=' ')
        # asaved = np.reshape(asaved, (224, 224, 3))
        # psaved = np.reshape(psaved, (224, 224, 3))
        # nsaved = np.reshape(nsaved, (224, 224, 3))
        # asaved = np.expand_dims(asaved, axis=0)
        # psaved = np.expand_dims(psaved, axis=0)
        # nsaved = np.expand_dims(nsaved, axis=0)
        # img_data = np.concatenate((asaved,psaved,nsaved),axis=0)

        with tf.variable_scope("") as scope:
            a_output,tt1 = self.CNN_Metric_Model(img_a)
            scope.reuse_variables()
            p_output,tt2 = self.CNN_Metric_Model(img_p)
            scope.reuse_variables()
            n_output, tt3= self.CNN_Metric_Model(img_n)

        a_output = tf.nn.l2_normalize(a_output,dim=1)
        p_output = tf.nn.l2_normalize(p_output,dim=1)
        n_output = tf.nn.l2_normalize(n_output,dim=1)

        loss,t1,t2,t3= self.triplet_loss([0.5,0.5,0.5],a_output,p_output,n_output)
        train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=self.var_dict)
        saver.restore(self.sess,"./inception_v1.ckpt")


        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./train', self.sess.graph)
        # for epoch in range(3):
        #     print self.sess.run(tf.get_default_graph().get_tensor_by_name("InceptionV1/Conv2d_1a_7x7/weights:0"))
        #     opt = self.sess.run([loss,train_op,tt1,tt2,tt3], feed_dict={img_a: asaved,
        #                                            img_p: psaved,
        #                                            img_n: nsaved})
        #     print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(opt[0]))
        #     print opt[2]
        # index_a, index_p, index_n = self.generate_triplet(class_label, 40)
        # opt = self.sess.run([loss,train_op], feed_dict={img_a: img_data[index_a, :, :, :],
        #                                                  img_p: img_data[index_p, :, :, :],
        #                                                  img_n: img_data[index_n, :, :, :]})
        #
        # print("Epoch:", '%04d' % (0 + 1), "cost=", "{:.9f}".format(opt[0]))

        for epoch in range(2000):
            # print self.sess.run(tf.get_default_graph().get_tensor_by_name("InceptionV1/Conv2d_1a_7x7/weights:0"))
            index_a, index_p, index_n = self.generate_triplet(class_label, 40)
            data_a,data_p,data_n = img_data[index_a,:,:,:],img_data[index_p,:,:,:],img_data[index_n,:,:,:]
            imean = np.tile(image_mean, (40, 1, 1, 1))
            data_a = np.float32(data_a) - np.float32(imean)
            data_p = np.float32(data_p) - np.float32(imean)
            data_n = np.float32(data_n) - np.float32(imean)


            cost,_,m = self.sess.run([loss,train_op,merged], feed_dict={img_a: data_a,
                                                             img_p: data_p,
                                                             img_n: data_n})
            print m
            train_writer.add_summary(m,epoch)
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost))

            # for epoch in range(19998):
            #     index_a, index_p, index_n = self.generate_triplet(class_label, 40)
            #
            #     scope.reuse_variables()
            #     a_output = self.CNN_Metric_Model(img_a, False)
            #     scope.reuse_variables()
            #     p_output = self.CNN_Metric_Model(img_p, False)
            #     scope.reuse_variables()
            #     n_output = self.CNN_Metric_Model(img_n, False)
            #
            #     loss, _, _, _ = self.triplet_loss([1.0, 1.0, 1.0], a_output, p_output, n_output)
            #     train_op = tf.train.AdamOptimizer(0.05).minimize(loss)
            #     opt = self.sess.run([loss,train_op], feed_dict={img_a: img_data[index_a, :, :, :],
            #                                            img_p: img_data[index_p, :, :, :],
            #                                            img_n: img_data[index_n, :, :, :]})
            #     print("Epoch:", '%04d' % (epoch + 2), "cost=", "{:.9f}".format(opt[0]))



        with h5py.File("validation_images_crop15_square256.mat") as f:
            img_data = [f[element[0]][:] for element in f['validation_images/img']]
            class_id = [f[element[0]][:] for element in f['validation_images/class_id']]

        img_data = np.asarray(img_data)
        class_id = np.asarray(class_id)
        img_data = np.transpose(img_data, (0, 2, 3, 1))
        class_label = class_id[:, 0, 0]

        for i in range(len(class_label) - 1):
            img_data[i, :, :, :] -= np.float32(image_mean)

        results = np.zeros((1, 64))

        ptr = 0
        no_of_batches_test = int(img_data.shape[0] / 100)
        for k in range(no_of_batches_test):
            inp = img_data[ptr:ptr + 100,:,:,:]
            imean = np.tile(image_mean, (100, 1, 1, 1))
            inp = np.float32(inp) - np.float32(imean)
            ptr += 100
            embeded_feat = self.sess.run([a_output], feed_dict={img_a: inp})
            results = np.concatenate((results,embeded_feat[0]), axis=0)

        results = np.delete(results, 0, axis=0)
        np.savetxt("results.csv", results, delimiter=",")
            # print results.shape

    def Variables_Dict(self):
        variables = {
            'InceptionV1/Conv2d_1a_7x7/weights':tf.get_variable(shape=[7,7,3,64],name='InceptionV1/Conv2d_1a_7x7/weights'),
            'InceptionV1/Conv2d_2b_1x1/weights': tf.get_variable(shape=[1,1,64,64], name='InceptionV1/Conv2d_2b_1x1/weights'),
            'InceptionV1/Conv2d_2c_3x3/weights': tf.get_variable(shape=[3,3,64,192], name='InceptionV1/Conv2d_2c_3x3/weights'),
            #first inception
            'InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,192,64], name='InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,192,96], name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(shape=[3,3,96,128], name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/weights'),
            'InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,192,16], name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/weights': tf.get_variable(shape=[3,3,16,32], name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/weights'),
            'InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(shape=[1,1,192,32], name='InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/weights'),
            #second inception
            'InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,256,128], name='InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,256,128], name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(shape=[3,3,128,192], name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/weights'),
            'InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,256,32], name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/weights': tf.get_variable(shape=[3,3,32,96], name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/weights'),
            'InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(shape=[1,1,256,64], name='InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/weights'),
            #third inception
            'InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,480,192], name='InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,480,96], name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(shape=[3,3,96,208], name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/weights'),
            'InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,480,16], name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/weights': tf.get_variable(shape=[3,3,16,48], name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/weights'),
            'InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(shape=[1,1,480,64], name='InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/weights'),
            # fourth inception
            'InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,512,160], name='InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,512,112], name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(shape=[3,3,112,224], name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/weights'),
            'InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,512,24], name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/weights': tf.get_variable(shape=[3,3,24,64], name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/weights'),
            'InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(shape=[1,1,512,64], name='InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/weights'),
            # fifth inception
            'InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,512,128], name='InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,512,128], name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(shape=[3,3,128,256], name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/weights'),
            'InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,512,24], name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/weights': tf.get_variable(shape=[3,3,24,64], name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/weights'),
            'InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(shape=[1,1,512,64], name='InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/weights'),
            # sixth inception
            'InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,512,112], name='InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,512,144], name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(shape=[3,3,144,288], name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/weights'),
            'InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,512,32], name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/weights': tf.get_variable(shape=[3,3,32,64], name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/weights'),
            'InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(shape=[1,1,512,64], name='InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/weights'),
            # seventh inception
            'InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,528,256], name='InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,528,160], name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(shape=[3,3,160,320], name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/weights'),
            'InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,528,32], name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/weights': tf.get_variable(shape=[3,3,32,128], name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/weights'),
            'InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(shape=[1,1,528,128], name='InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/weights'),
            # eighth inception
            'InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,832,256], name='InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,832,160], name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(shape=[3,3,160,320], name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/weights'),
            'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,832,32], name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/weights': tf.get_variable(shape=[3,3,32,128], name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/weights'),
            'InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(shape=[1,1,832,128], name='InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/weights'),
            # ninth inception
            'InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,832,384], name='InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,832,192], name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(shape=[3,3,192,384], name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/weights'),
            'InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(shape=[1,1,832,48], name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/weights'),
            'InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/weights': tf.get_variable(shape=[3,3,48,128], name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/weights'),
            'InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(shape=[1,1,832,128], name='InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/weights'),
            #beta
            'InceptionV1/Conv2d_1a_7x7/BatchNorm/beta':tf.get_variable(shape=[64],name='InceptionV1/Conv2d_1a_7x7/BatchNorm/beta'),
            'InceptionV1/Conv2d_2b_1x1/BatchNorm/beta':tf.get_variable(shape=[64],name='InceptionV1/Conv2d_2b_1x1/BatchNorm/beta'),
            'InceptionV1/Conv2d_2c_3x3/BatchNorm/beta': tf.get_variable(shape=[192],name='InceptionV1/Conv2d_2c_3x3/BatchNorm/beta'),
            #first inception
            'InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[64],name='InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[96],name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[128],name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[16],name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[32],name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[32],name='InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta'),
            # second inception
            'InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[128],name='InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[128],name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[192],name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[32],name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[96],name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[64],name='InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta'),
            # third inception
            'InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[192],name='InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[96],name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[208],name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[16],name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[48],name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[64],name='InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta'),
            # fourth inception
            'InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[160],name='InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[112],name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[224],name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[24],name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[64],name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[64],name='InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta'),
            # fifth inception
            'InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[128],name='InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[128],name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[256],name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[24],name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[64],name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[64],name='InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/beta'),
            # sixth inception
            'InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[112],name='InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[144],name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[288],name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[32],name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[64],name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[64],name='InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/beta'),
            # seventh inception
            'InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[256],name='InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[160],name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[320],name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[32],name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[128],name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[128],name='InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/beta'),
            # eighth inception
            'InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[256],name='InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[160],name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[320],name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[32],name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/beta': tf.get_variable(shape=[128],name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[128],name='InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta'),
            # ninth inception
            'InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[384],name='InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[192],name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[384],name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[48],name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta'),
            'InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[128],name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta'),
            'InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[128],name='InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta'),
            #moving means
            'InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_mean': tf.get_variable(shape=[64], name='InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_mean',trainable=False),
            'InceptionV1/Conv2d_2b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[64], name='InceptionV1/Conv2d_2b_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Conv2d_2c_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[192], name='InceptionV1/Conv2d_2c_3x3/BatchNorm/moving_mean'),
            # first inception
            'InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[64], name='InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[96], name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[128], name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[16], name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[32], name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[32], name='InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean'),
            # second inception
            'InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[128], name='InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[128], name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[192], name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[32], name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[96], name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[64], name='InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean'),
            # third inception
            'InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[192], name='InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[96], name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[208], name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[16], name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[48], name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[64],name='InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean'),
            # fourth inception
            'InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[160], name='InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[112], name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[224], name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[24], name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[64], name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[64], name='InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean'),
            # fifth inception
            'InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[128], name='InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[128], name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[256], name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[24],  name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[64], name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[64], name='InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean'),
            # sixth inception
            'InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[112], name='InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[144], name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[288], name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[32],  name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[64], name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[64], name='InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean'),
            # seventh inception
            'InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[256], name='InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[160], name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[320], name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[32], name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[128],  name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[128], name='InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean'),
            # eighth inception
            'InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[256], name='InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[160], name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[320], name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[32], name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[128],  name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[128], name='InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean'),
            # ninth inception
            'InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[384], name='InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[192],  name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[384], name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[48],  name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[128], name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean'),
            'InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[128], name='InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean'),
            #moving variance
            'InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_variance': tf.get_variable(shape=[64], name='InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_variance'),
            'InceptionV1/Conv2d_2b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[64], name='InceptionV1/Conv2d_2b_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Conv2d_2c_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[192], name='InceptionV1/Conv2d_2c_3x3/BatchNorm/moving_variance'),
            # first inception
            'InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[64], name='InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[96], name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[128], name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[16], name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[32], name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[32], name='InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance'),
            # second inception
            'InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[128], name='InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[128], name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[192], name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[32], name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[96], name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[64], name='InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance'),
            # third inception
            'InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[192], name='InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[96],  name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[208], name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[16], name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[48], name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[64], name='InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance'),
            # fourth inception
            'InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[160], name='InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[112], name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[224], name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[24], name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[64], name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[64], name='InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance'),
            # fifth inception
            'InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[128], name='InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[128], name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[256],  name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[24], name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[64], name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[64], name='InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance'),
            # sixth inception
            'InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[112], name='InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[144], name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[288],  name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[32],  name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[64], name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[64], name='InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance'),
            # seventh inception
            'InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[256], name='InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[160], name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[320], name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[32], name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[128], name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[128], name='InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance'),
            # eighth inception
            'InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[256], name='InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[160],  name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[320],name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[32],  name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[128], name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[128], name='InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance'),
            # ninth inception
            'InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[384], name='InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[192], name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[384], name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[48], name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[128], name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance'),
            'InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[128],  name='InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance')
            # # beta
            # 'InceptionV1/Conv2d_1a_7x7/BatchNorm/beta': tf.get_variable(shape=[64],
            #                                                             name='InceptionV1/Conv2d_1a_7x7/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Conv2d_2b_1x1/BatchNorm/beta': tf.get_variable(shape=[64],
            #                                                             name='InceptionV1/Conv2d_2b_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Conv2d_2c_3x3/BatchNorm/beta': tf.get_variable(shape=[192],
            #                                                             name='InceptionV1/Conv2d_2c_3x3/BatchNorm/beta',trainable=False),
            # # first inception
            # 'InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[64],
            #                                                                               name='InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[96],
            #                                                                               name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[128],
            #                                                                               name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[16],
            #                                                                               name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[32],
            #                                                                               name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[32],
            #                                                                               name='InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',trainable=False),
            # # second inception
            # 'InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[128],
            #                                                                               name='InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[128],
            #                                                                               name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[192],
            #                                                                               name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[32],
            #                                                                               name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[96],
            #                                                                               name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[64],
            #                                                                               name='InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',trainable=False),
            # # third inception
            # 'InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[192],
            #                                                                               name='InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[96],
            #                                                                               name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[208],
            #                                                                               name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[16],
            #                                                                               name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[48],
            #                                                                               name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[64],
            #                                                                               name='InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',trainable=False),
            # # fourth inception
            # 'InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[160],
            #                                                                               name='InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[112],
            #                                                                               name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[224],
            #                                                                               name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[24],
            #                                                                               name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[64],
            #                                                                               name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[64],
            #                                                                               name='InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',trainable=False),
            # # fifth inception
            # 'InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[128],
            #                                                                               name='InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[128],
            #                                                                               name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[256],
            #                                                                               name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[24],
            #                                                                               name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[64],
            #                                                                               name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[64],
            #                                                                               name='InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',trainable=False),
            # # sixth inception
            # 'InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[112],
            #                                                                               name='InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[144],
            #                                                                               name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[288],
            #                                                                               name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[32],
            #                                                                               name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[64],
            #                                                                               name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[64],
            #                                                                               name='InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',trainable=False),
            # # seventh inception
            # 'InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[256],
            #                                                                               name='InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[160],
            #                                                                               name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[320],
            #                                                                               name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[32],
            #                                                                               name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[128],
            #                                                                               name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[128],
            #                                                                               name='InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',trainable=False),
            # # eighth inception
            # 'InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[256],
            #                                                                               name='InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[160],
            #                                                                               name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[320],
            #                                                                               name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[32],
            #                                                                               name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/beta': tf.get_variable(shape=[128],
            #                                                                               name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[128],
            #                                                                               name='InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',trainable=False),
            # # ninth inception
            # 'InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[384],
            #                                                                               name='InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[192],
            #                                                                               name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[384],
            #                                                                               name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta': tf.get_variable(shape=[48],
            #                                                                               name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta': tf.get_variable(shape=[128],
            #                                                                               name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta',trainable=False),
            # 'InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta': tf.get_variable(shape=[128],
            #                                                                               name='InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',trainable=False),
            # # moving means
            # 'InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_mean': tf.get_variable(shape=[64],
            #                                                                    name='InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_mean',
            #                                                                    trainable=False),
            # 'InceptionV1/Conv2d_2b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[64],
            #                                                                    name='InceptionV1/Conv2d_2b_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Conv2d_2c_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[192],
            #                                                                    name='InceptionV1/Conv2d_2c_3x3/BatchNorm/moving_mean',trainable=False),
            # # first inception
            # 'InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[64],
            #                                                                                      name='InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[96],
            #                                                                                      name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[128],
            #                                                                                      name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[16],
            #                                                                                      name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[32],
            #                                                                                      name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[32],
            #                                                                                      name='InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',trainable=False),
            # # second inception
            # 'InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[128],
            #                                                                                      name='InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[128],
            #                                                                                      name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[192],
            #                                                                                      name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[32],
            #                                                                                      name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[96],
            #                                                                                      name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[64],
            #                                                                                      name='InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',trainable=False),
            # # third inception
            # 'InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[192],
            #                                                                                      name='InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[96],
            #                                                                                      name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[208],
            #                                                                                      name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[16],
            #                                                                                      name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[48],
            #                                                                                      name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[64],
            #                                                                                      name='InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',trainable=False),
            # # fourth inception
            # 'InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[160],
            #                                                                                      name='InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[112],
            #                                                                                      name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[224],
            #                                                                                      name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[24],
            #                                                                                      name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[64],
            #                                                                                      name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[64],
            #                                                                                      name='InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',trainable=False),
            # # fifth inception
            # 'InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[128],
            #                                                                                      name='InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[128],
            #                                                                                      name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[256],
            #                                                                                      name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[24],
            #                                                                                      name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[64],
            #                                                                                      name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[64],
            #                                                                                      name='InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',trainable=False),
            # # sixth inception
            # 'InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[112],
            #                                                                                      name='InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[144],
            #                                                                                      name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[288],
            #                                                                                      name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[32],
            #                                                                                      name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[64],
            #                                                                                      name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[64],
            #                                                                                      name='InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',trainable=False),
            # # seventh inception
            # 'InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[256],
            #                                                                                      name='InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[160],
            #                                                                                      name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[320],
            #                                                                                      name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[32],
            #                                                                                      name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[128],
            #                                                                                      name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[128],
            #                                                                                      name='InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',trainable=False),
            # # eighth inception
            # 'InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[256],
            #                                                                                      name='InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[160],
            #                                                                                      name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[320],
            #                                                                                      name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[32],
            #                                                                                      name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[128],
            #                                                                                      name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[128],
            #                                                                                      name='InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',trainable=False),
            # # ninth inception
            # 'InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[384],
            #                                                                                      name='InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[192],
            #                                                                                      name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[384],
            #                                                                                      name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[48],
            #                                                                                      name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean': tf.get_variable(shape=[128],
            #                                                                                      name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean',trainable=False),
            # 'InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean': tf.get_variable(shape=[128],
            #                                                                                      name='InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',trainable=False),
            # # moving variance
            # 'InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_variance': tf.get_variable(shape=[64],
            #                                                                        name='InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Conv2d_2b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[64],
            #                                                                        name='InceptionV1/Conv2d_2b_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Conv2d_2c_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[192],
            #                                                                        name='InceptionV1/Conv2d_2c_3x3/BatchNorm/moving_variance',trainable=False),
            # # first inception
            # 'InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[64],
            #                                                                                          name='InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[96],
            #                                                                                          name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[128],
            #                                                                                          name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[16],
            #                                                                                          name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[32],
            #                                                                                          name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[32],
            #                                                                                          name='InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',trainable=False),
            # # second inception
            # 'InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[128],
            #                                                                                          name='InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[128],
            #                                                                                          name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[192],
            #                                                                                          name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[32],
            #                                                                                          name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[96],
            #                                                                                          name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[64],
            #                                                                                          name='InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',trainable=False),
            # # third inception
            # 'InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[192],
            #                                                                                          name='InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[96],
            #                                                                                          name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[208],
            #                                                                                          name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[16],
            #                                                                                          name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[48],
            #                                                                                          name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[64],
            #                                                                                          name='InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',trainable=False),
            # # fourth inception
            # 'InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[160],
            #                                                                                          name='InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[112],
            #                                                                                          name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[224],
            #                                                                                          name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[24],
            #                                                                                          name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[64],
            #                                                                                          name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[64],
            #                                                                                          name='InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',trainable=False),
            # # fifth inception
            # 'InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[128],
            #                                                                                          name='InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[128],
            #                                                                                          name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[256],
            #                                                                                          name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[24],
            #                                                                                          name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[64],
            #                                                                                          name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[64],
            #                                                                                          name='InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',trainable=False),
            # # sixth inception
            # 'InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[112],
            #                                                                                          name='InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[144],
            #                                                                                          name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[288],
            #                                                                                          name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[32],
            #                                                                                          name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[64],
            #                                                                                          name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[64],
            #                                                                                          name='InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',trainable=False),
            # # seventh inception
            # 'InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[256],
            #                                                                                          name='InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[160],
            #                                                                                          name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[320],
            #                                                                                          name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[32],
            #                                                                                          name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[128],
            #                                                                                          name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[128],
            #                                                                                          name='InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',trainable=False),
            # # eighth inception
            # 'InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[256],
            #                                                                                          name='InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[160],
            #                                                                                          name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[320],
            #                                                                                          name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[32],
            #                                                                                          name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[128],
            #                                                                                          name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[128],
            #                                                                                          name='InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',trainable=False),
            # # ninth inception
            # 'InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[384],
            #                                                                                          name='InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[192],
            #                                                                                          name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[384],
            #                                                                                          name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[48],
            #                                                                                          name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance': tf.get_variable(shape=[128],
            #                                                                                          name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance',trainable=False),
            # 'InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance': tf.get_variable(shape=[128],
            #                                                                                          name='InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',trainable=False)

        }
        return variables
    def CNN_Metric_Model(self,x):
        #layer 1 - conv
        w_1 = self.var_dict['InceptionV1/Conv2d_1a_7x7/weights']
        h_conv1 = tf.nn.conv2d(x, w_1, strides=[1, 2, 2, 1], padding='SAME')
        h_conv1 = tf.nn.relu(h_conv1)
        #layer 1 - max pool
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1],\
                                 strides=[1, 2, 2, 1], padding='SAME')

        #layer 1 -  BN
        beta_1 = self.var_dict['InceptionV1/Conv2d_1a_7x7/BatchNorm/beta']
        moving_mean_1 = self.var_dict['InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_mean']
        moving_variance_1 = self.var_dict['InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_variance']
        # h_bn1 = tf.nn.batch_normalization(h_pool1,offset=beta_1,\
        #                                       mean=moving_mean_1, \
        #                                       variance=moving_variance_1,scale=None,variance_epsilon=0.1)
        h_bn1 = tf.add(tf.div(tf.subtract(h_pool1, moving_mean_1),tf.add(moving_variance_1,tf.constant(0.01))),beta_1)
        test = h_bn1
        #layer 2 - conv
        w_2 = self.var_dict['InceptionV1/Conv2d_2b_1x1/weights']
        h_conv2 = tf.nn.conv2d(h_bn1, w_2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2 = tf.nn.relu(h_conv2)
        #layer 2 -  BN
        beta_2 = self.var_dict['InceptionV1/Conv2d_2b_1x1/BatchNorm/beta']
        moving_mean_2 = self.var_dict['InceptionV1/Conv2d_2b_1x1/BatchNorm/moving_mean']
        moving_variance_2 = self.var_dict['InceptionV1/Conv2d_2b_1x1/BatchNorm/moving_variance']
        h_bn2 = tf.nn.batch_normalization(h_conv2,offset=beta_2,\
                                              mean=moving_mean_2, \
                                              variance=moving_variance_2,scale=None,variance_epsilon=0.001)
        #layer 3 - conv
        w_3 = self.var_dict['InceptionV1/Conv2d_2c_3x3/weights']
        h_conv3 = tf.nn.conv2d(h_bn2, w_3, strides=[1, 1, 1, 1], padding='SAME')
        h_conv3 = tf.nn.relu(h_conv3)
        #layer 3 - BN
        beta_3 = self.var_dict['InceptionV1/Conv2d_2c_3x3/BatchNorm/beta']
        moving_mean_3 = self.var_dict['InceptionV1/Conv2d_2c_3x3/BatchNorm/moving_mean']
        moving_variance_3 = self.var_dict['InceptionV1/Conv2d_2c_3x3/BatchNorm/moving_variance']
        h_bn3 = tf.nn.batch_normalization(h_conv3,offset=beta_3,\
                                              mean=moving_mean_3, \
                                              variance=moving_variance_3,scale=None,variance_epsilon=0.001)
        #layer 3 - max pool
        h_pool3 = tf.nn.max_pool(h_bn3, ksize=[1, 3, 3, 1], \
                                 strides=[1, 2, 2, 1], padding='SAME')
        #mixed layer 3b
        #first inception
        #branch 0
        w_4 = self.var_dict['InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/weights']
        branch1_0 = tf.nn.conv2d(h_pool3, w_4, strides=[1, 1, 1, 1], padding='SAME')
        branch1_0 = tf.nn.relu(branch1_0)
        beta_4 = self.var_dict['InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_4 = self.var_dict['InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_4 = self.var_dict['InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch1_0 = tf.nn.batch_normalization(branch1_0,offset=beta_4,\
                                              mean=moving_mean_4, \
                                              variance=moving_variance_4,scale=None,variance_epsilon=0.001)
        #branch 1
        w_5 = self.var_dict['InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/weights']
        branch1_1 = tf.nn.conv2d(h_pool3, w_5, strides=[1, 1, 1, 1], padding='SAME')
        branch1_1 = tf.nn.relu(branch1_1)
        beta_5 = self.var_dict['InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_5 = self.var_dict['InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_5 = self.var_dict['InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch1_1 = tf.nn.batch_normalization(branch1_1,offset=beta_5,\
                                              mean=moving_mean_5, \
                                              variance=moving_variance_5,scale=None,variance_epsilon=0.001)
        w_6 = self.var_dict['InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/weights']
        branch1_1 = tf.nn.conv2d(branch1_1, w_6, strides=[1, 1, 1, 1], padding='SAME')
        branch1_1 = tf.nn.relu(branch1_1)
        beta_6 = self.var_dict['InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta']
        moving_mean_6 = self.var_dict['InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean']
        moving_variance_6 = self.var_dict['InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance']
        branch1_1 = tf.nn.batch_normalization(branch1_1,offset=beta_6,\
                                              mean=moving_mean_6, \
                                              variance=moving_variance_6,scale=None,variance_epsilon=0.001)
        #branch 2
        w_7 = self.var_dict['InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/weights']
        branch1_2 = tf.nn.conv2d(h_pool3, w_7, strides=[1, 1, 1, 1], padding='SAME')
        branch1_2 = tf.nn.relu(branch1_2)
        beta_7 = self.var_dict['InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_7 = self.var_dict['InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_7 = self.var_dict['InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch1_2 = tf.nn.batch_normalization(branch1_2,offset=beta_7,\
                                              mean=moving_mean_7, \
                                              variance=moving_variance_7,scale=None,variance_epsilon=0.001)
        w_8 = self.var_dict['InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/weights']
        branch1_2 = tf.nn.conv2d(branch1_2, w_8, strides=[1, 1, 1, 1], padding='SAME')
        branch1_2 = tf.nn.relu(branch1_2)
        beta_8 = self.var_dict['InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/beta']
        moving_mean_8 = self.var_dict['InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean']
        moving_variance_8 = self.var_dict['InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance']
        branch1_2 = tf.nn.batch_normalization(branch1_2,offset=beta_8,\
                                              mean=moving_mean_8, \
                                              variance=moving_variance_8,scale=None,variance_epsilon=0.001)
        #branch 3
        branch1_3 = tf.nn.max_pool(h_pool3, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_9 = self.var_dict['InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/weights']
        branch1_3 = tf.nn.conv2d(branch1_3, w_9, strides=[1, 1, 1, 1], padding='SAME')
        branch1_3 = tf.nn.relu(branch1_3)
        beta_9 = self.var_dict['InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta']
        moving_mean_9 = self.var_dict['InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean']
        moving_variance_9 = self.var_dict['InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance']
        branch1_3 = tf.nn.batch_normalization(branch1_3,offset=beta_9,\
                                              mean=moving_mean_9, \
                                              variance=moving_variance_9,scale=None,variance_epsilon=0.001)
        incpt = tf.concat(
            axis=3, values=[branch1_0, branch1_1, branch1_2, branch1_3])
        #second inception
        #branch 0
        w_10 = self.var_dict['InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/weights']
        branch2_0 = tf.nn.conv2d(incpt, w_10, strides=[1, 1, 1, 1], padding='SAME')
        branch2_0 = tf.nn.relu(branch2_0)
        beta_10 = self.var_dict['InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_10 = self.var_dict['InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_10 = self.var_dict['InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch2_0 = tf.nn.batch_normalization(branch2_0,offset=beta_10,\
                                              mean=moving_mean_10, \
                                              variance=moving_variance_10,scale=None,variance_epsilon=0.001)
        #branch 1
        w_11 = self.var_dict['InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/weights']
        branch2_1 = tf.nn.conv2d(incpt, w_11, strides=[1, 1, 1, 1], padding='SAME')
        branch2_1 = tf.nn.relu(branch2_1)
        beta_11 = self.var_dict['InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_11 = self.var_dict['InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_11 = self.var_dict['InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch2_1 = tf.nn.batch_normalization(branch2_1,offset=beta_11,\
                                              mean=moving_mean_11, \
                                              variance=moving_variance_11,scale=None,variance_epsilon=0.001)
        w_12 = self.var_dict['InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/weights']
        branch2_1 = tf.nn.conv2d(branch2_1, w_12, strides=[1, 1, 1, 1], padding='SAME')
        branch2_1 = tf.nn.relu(branch2_1)
        beta_12 = self.var_dict['InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta']
        moving_mean_12 = self.var_dict['InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean']
        moving_variance_12 = self.var_dict['InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance']
        branch2_1 = tf.nn.batch_normalization(branch2_1,offset=beta_12,\
                                              mean=moving_mean_12, \
                                              variance=moving_variance_12,scale=None,variance_epsilon=0.001)
        #branch 2
        w_13 = self.var_dict['InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/weights']
        branch2_2 = tf.nn.conv2d(incpt, w_13, strides=[1, 1, 1, 1], padding='SAME')
        branch2_2 = tf.nn.relu(branch2_2)
        beta_13 = self.var_dict['InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_13 = self.var_dict['InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_13 = self.var_dict['InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch2_2 = tf.nn.batch_normalization(branch2_2,offset=beta_13,\
                                              mean=moving_mean_13, \
                                              variance=moving_variance_13,scale=None,variance_epsilon=0.001)
        w_14 = self.var_dict['InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/weights']
        branch2_2 = tf.nn.conv2d(branch2_2, w_14, strides=[1, 1, 1, 1], padding='SAME')
        branch2_2 = tf.nn.relu(branch2_2)
        beta_14 = self.var_dict['InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta']
        moving_mean_14 = self.var_dict['InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean']
        moving_variance_14 = self.var_dict['InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance']
        branch2_2 = tf.nn.batch_normalization(branch2_2,offset=beta_14,\
                                              mean=moving_mean_14, \
                                              variance=moving_variance_14,scale=None,variance_epsilon=0.001)
        #branch 3
        branch2_3 = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_15 = self.var_dict['InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/weights']
        branch2_2 = tf.nn.relu(branch2_2)
        branch2_3 = tf.nn.conv2d(branch2_3, w_15, strides=[1, 1, 1, 1], padding='SAME')
        beta_15 = self.var_dict['InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta']
        moving_mean_15 = self.var_dict['InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean']
        moving_variance_15 = self.var_dict['InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance']
        branch2_3 = tf.nn.batch_normalization(branch2_3,offset=beta_15,\
                                              mean=moving_mean_15, \
                                              variance=moving_variance_15,scale=None,variance_epsilon=0.001)
        incpt = tf.concat(
            axis=3, values=[branch2_0, branch2_1, branch2_2, branch2_3])
        incpt = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                   strides=[1, 2, 2, 1], padding='SAME')
        #third inception
        #branch 0
        w_16 = self.var_dict['InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/weights']
        branch3_0 = tf.nn.conv2d(incpt, w_16, strides=[1, 1, 1, 1], padding='SAME')
        branch3_0 = tf.nn.relu(branch3_0)
        beta_16 = self.var_dict['InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_16 = self.var_dict['InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_16 = self.var_dict['InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch3_0 = tf.nn.batch_normalization(branch3_0,offset=beta_16,\
                                              mean=moving_mean_16, \
                                              variance=moving_variance_16,scale=None,variance_epsilon=0.001)
        #branch 1
        w_17 = self.var_dict['InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/weights']
        branch3_1 = tf.nn.conv2d(incpt, w_17, strides=[1, 1, 1, 1], padding='SAME')
        branch3_1 = tf.nn.relu(branch3_1)
        beta_17 = self.var_dict['InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_17 = self.var_dict['InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_17 = self.var_dict['InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch3_1 = tf.nn.batch_normalization(branch3_1,offset=beta_17,\
                                              mean=moving_mean_17, \
                                              variance=moving_variance_17,scale=None,variance_epsilon=0.001)
        w_18 = self.var_dict['InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/weights']
        branch3_1 = tf.nn.conv2d(branch3_1, w_18, strides=[1, 1, 1, 1], padding='SAME')
        branch3_1 = tf.nn.relu(branch3_1)
        beta_18 = self.var_dict['InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta']
        moving_mean_18 = self.var_dict['InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean']
        moving_variance_18 = self.var_dict['InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance']
        branch3_1 = tf.nn.batch_normalization(branch3_1,offset=beta_18,\
                                              mean=moving_mean_18, \
                                              variance=moving_variance_18,scale=None,variance_epsilon=0.001)
        #branch 2
        w_19 = self.var_dict['InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/weights']
        branch3_2 = tf.nn.conv2d(incpt, w_19, strides=[1, 1, 1, 1], padding='SAME')
        branch3_2 = tf.nn.relu(branch3_2)
        beta_19 = self.var_dict['InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_19 = self.var_dict['InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_19 = self.var_dict['InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch3_2 = tf.nn.batch_normalization(branch3_2,offset=beta_19,\
                                              mean=moving_mean_19, \
                                              variance=moving_variance_19,scale=None,variance_epsilon=0.001)
        w_20 = self.var_dict['InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/weights']
        branch3_2 = tf.nn.conv2d(branch3_2, w_20, strides=[1, 1, 1, 1], padding='SAME')
        branch3_2 = tf.nn.relu(branch3_2)
        beta_20 = self.var_dict['InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/beta']
        moving_mean_20 = self.var_dict['InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean']
        moving_variance_20 = self.var_dict['InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance']
        branch3_2 = tf.nn.batch_normalization(branch3_2,offset=beta_20,\
                                              mean=moving_mean_20, \
                                              variance=moving_variance_20,scale=None,variance_epsilon=0.001)
        #branch 3
        branch3_3 = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_21 = self.var_dict['InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/weights']
        branch3_3 = tf.nn.conv2d(branch3_3, w_21, strides=[1, 1, 1, 1], padding='SAME')
        branch3_3 = tf.nn.relu(branch3_3)
        beta_21 = self.var_dict['InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta']
        moving_mean_21 = self.var_dict['InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean']
        moving_variance_21 = self.var_dict['InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance']
        branch3_3 = tf.nn.batch_normalization(branch3_3,offset=beta_21,\
                                              mean=moving_mean_21, \
                                              variance=moving_variance_21,scale=None,variance_epsilon=0.001)
        incpt = tf.concat(
            axis=3, values=[branch3_0, branch3_1, branch3_2, branch3_3])
        #fourth inception
        #branch 0
        w_22 = self.var_dict['InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/weights']
        branch4_0 = tf.nn.conv2d(incpt, w_22, strides=[1, 1, 1, 1], padding='SAME')
        branch4_0 = tf.nn.relu(branch4_0)
        beta_22 = self.var_dict['InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_22 = self.var_dict['InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_22 = self.var_dict['InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch4_0 = tf.nn.batch_normalization(branch4_0,offset=beta_22,\
                                              mean=moving_mean_22, \
                                              variance=moving_variance_22,scale=None,variance_epsilon=0.001)
        #branch 1
        w_23 = self.var_dict['InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/weights']
        branch4_1 = tf.nn.conv2d(incpt, w_23, strides=[1, 1, 1, 1], padding='SAME')
        branch4_1 = tf.nn.relu(branch4_1)
        beta_23 = self.var_dict['InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_23 = self.var_dict['InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_23 = self.var_dict['InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch4_1 = tf.nn.batch_normalization(branch4_1,offset=beta_23,\
                                              mean=moving_mean_23, \
                                              variance=moving_variance_23,scale=None,variance_epsilon=0.001)
        w_24 = self.var_dict['InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/weights']
        branch4_1 = tf.nn.conv2d(branch4_1, w_24, strides=[1, 1, 1, 1], padding='SAME')
        branch4_1 = tf.nn.relu(branch4_1)
        beta_24 = self.var_dict['InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta']
        moving_mean_24 = self.var_dict['InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean']
        moving_variance_24 = self.var_dict['InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance']
        branch4_1 = tf.nn.batch_normalization(branch4_1,offset=beta_24,\
                                              mean=moving_mean_24, \
                                              variance=moving_variance_24,scale=None,variance_epsilon=0.001)
        #branch 2
        w_25 = self.var_dict['InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/weights']
        branch4_2 = tf.nn.conv2d(incpt, w_25, strides=[1, 1, 1, 1], padding='SAME')
        branch4_2 = tf.nn.relu(branch4_2)
        beta_25 = self.var_dict['InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_25 = self.var_dict['InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_25 = self.var_dict['InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch4_2 = tf.nn.batch_normalization(branch4_2,offset=beta_25,\
                                              mean=moving_mean_25, \
                                              variance=moving_variance_25,scale=None,variance_epsilon=0.001)
        w_26 = self.var_dict['InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/weights']
        branch4_2 = tf.nn.conv2d(branch4_2, w_26, strides=[1, 1, 1, 1], padding='SAME')
        branch4_2 = tf.nn.relu(branch4_2)
        beta_26 = self.var_dict['InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta']
        moving_mean_26 = self.var_dict['InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean']
        moving_variance_26 = self.var_dict['InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance']
        branch4_2 = tf.nn.batch_normalization(branch4_2,offset=beta_26,\
                                              mean=moving_mean_26, \
                                              variance=moving_variance_26,scale=None,variance_epsilon=0.001)
        #branch 3
        branch4_3 = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_27 = self.var_dict['InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/weights']
        branch4_3 = tf.nn.conv2d(branch4_3, w_27, strides=[1, 1, 1, 1], padding='SAME')
        branch4_3 = tf.nn.relu(branch4_3)
        beta_27 = self.var_dict['InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta']
        moving_mean_27 = self.var_dict['InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean']
        moving_variance_27 = self.var_dict['InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance']
        branch4_3 = tf.nn.batch_normalization(branch4_3,offset=beta_27,\
                                              mean=moving_mean_27, \
                                              variance=moving_variance_27,scale=None,variance_epsilon=0.001)
        incpt = tf.concat(
            axis=3, values=[branch4_0, branch4_1, branch4_2, branch4_3])
        #fifth inception
        #branch 0
        w_28 = self.var_dict['InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/weights']
        branch5_0 = tf.nn.conv2d(incpt, w_28, strides=[1, 1, 1, 1], padding='SAME')
        branch5_0 = tf.nn.relu(branch5_0)
        beta_28 = self.var_dict['InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_28 = self.var_dict['InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_28 = self.var_dict['InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch5_0 = tf.nn.batch_normalization(branch5_0,offset=beta_28,\
                                              mean=moving_mean_28, \
                                              variance=moving_variance_28,scale=None,variance_epsilon=0.001)
        #branch 1
        w_29 = self.var_dict['InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/weights']
        branch5_1 = tf.nn.conv2d(incpt, w_29, strides=[1, 1, 1, 1], padding='SAME')
        branch5_1 = tf.nn.relu(branch5_1)
        beta_29 = self.var_dict['InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_29 = self.var_dict['InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_29 = self.var_dict['InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch5_1 = tf.nn.batch_normalization(branch5_1,offset=beta_29,\
                                              mean=moving_mean_29, \
                                              variance=moving_variance_29,scale=None,variance_epsilon=0.001)
        w_30 = self.var_dict['InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/weights']
        branch5_1 = tf.nn.conv2d(branch5_1, w_30, strides=[1, 1, 1, 1], padding='SAME')
        branch5_1 = tf.nn.relu(branch5_1)
        beta_30 = self.var_dict['InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/beta']
        moving_mean_30 = self.var_dict['InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean']
        moving_variance_30 = self.var_dict['InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance']
        branch5_1 = tf.nn.batch_normalization(branch5_1,offset=beta_30,\
                                              mean=moving_mean_30, \
                                              variance=moving_variance_30,scale=None,variance_epsilon=0.001)
        #branch 2
        w_31 = self.var_dict['InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/weights']
        branch5_2 = tf.nn.conv2d(incpt, w_31, strides=[1, 1, 1, 1], padding='SAME')
        branch5_2 = tf.nn.relu(branch5_2)
        beta_31 = self.var_dict['InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_31 = self.var_dict['InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_31 = self.var_dict['InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch5_2 = tf.nn.batch_normalization(branch5_2,offset=beta_31,\
                                              mean=moving_mean_31, \
                                              variance=moving_variance_31,scale=None,variance_epsilon=0.001)
        w_32 = self.var_dict['InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/weights']
        branch5_2 = tf.nn.conv2d(branch5_2, w_32, strides=[1, 1, 1, 1], padding='SAME')
        branch5_2 = tf.nn.relu(branch5_2)
        beta_32 = self.var_dict['InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/beta']
        moving_mean_32 = self.var_dict['InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean']
        moving_variance_32 = self.var_dict['InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance']
        branch5_2 = tf.nn.batch_normalization(branch5_2,offset=beta_32,\
                                              mean=moving_mean_32, \
                                              variance=moving_variance_32,scale=None,variance_epsilon=0.001)
        #branch 3
        branch5_3 = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_33 = self.var_dict['InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/weights']
        branch5_3 = tf.nn.conv2d(branch5_3, w_33, strides=[1, 1, 1, 1], padding='SAME')
        branch5_3 = tf.nn.relu(branch5_3)
        beta_33 = self.var_dict['InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/beta']
        moving_mean_33 = self.var_dict['InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean']
        moving_variance_33 = self.var_dict['InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance']
        branch5_3 = tf.nn.batch_normalization(branch5_3,offset=beta_33,\
                                              mean=moving_mean_33, \
                                              variance=moving_variance_33,scale=None,variance_epsilon=0.001)
        incpt = tf.concat(
            axis=3, values=[branch5_0, branch5_1, branch5_2, branch5_3])
        #sixth inception
        #branch 0
        w_34 = self.var_dict['InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/weights']
        branch6_0 = tf.nn.conv2d(incpt, w_34, strides=[1, 1, 1, 1], padding='SAME')
        branch6_0 = tf.nn.relu(branch6_0)
        beta_34 = self.var_dict['InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_34 = self.var_dict['InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_34 = self.var_dict['InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch6_0 = tf.nn.batch_normalization(branch6_0,offset=beta_34,\
                                              mean=moving_mean_34, \
                                              variance=moving_variance_34,scale=None,variance_epsilon=0.001)
        #branch 1
        w_35 = self.var_dict['InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/weights']
        branch6_1 = tf.nn.conv2d(incpt, w_35, strides=[1, 1, 1, 1], padding='SAME')
        branch6_1 = tf.nn.relu(branch6_1)
        beta_35 = self.var_dict['InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_35 = self.var_dict['InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_35 = self.var_dict['InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch6_1 = tf.nn.batch_normalization(branch6_1,offset=beta_35,\
                                              mean=moving_mean_35, \
                                              variance=moving_variance_35,scale=None,variance_epsilon=0.001)
        w_36 = self.var_dict['InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/weights']
        branch6_1 = tf.nn.conv2d(branch6_1, w_36, strides=[1, 1, 1, 1], padding='SAME')
        branch6_1 = tf.nn.relu(branch6_1)
        beta_36 = self.var_dict['InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/beta']
        moving_mean_36 = self.var_dict['InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean']
        moving_variance_36 = self.var_dict['InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance']
        branch6_1 = tf.nn.batch_normalization(branch6_1,offset=beta_36,\
                                              mean=moving_mean_36, \
                                              variance=moving_variance_36,scale=None,variance_epsilon=0.001)
        #branch 2
        w_37 = self.var_dict['InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/weights']
        branch6_2 = tf.nn.conv2d(incpt, w_37, strides=[1, 1, 1, 1], padding='SAME')
        branch6_2 = tf.nn.relu(branch6_2)
        beta_37 = self.var_dict['InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_37 = self.var_dict['InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_37 = self.var_dict['InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch6_2 = tf.nn.batch_normalization(branch6_2,offset=beta_37,\
                                              mean=moving_mean_37, \
                                              variance=moving_variance_37,scale=None,variance_epsilon=0.001)
        w_38 = self.var_dict['InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/weights']
        branch6_2 = tf.nn.conv2d(branch6_2, w_38, strides=[1, 1, 1, 1], padding='SAME')
        branch6_2 = tf.nn.relu(branch6_2)
        beta_38 = self.var_dict['InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/beta']
        moving_mean_38 = self.var_dict['InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean']
        moving_variance_38 = self.var_dict['InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance']
        branch6_2 = tf.nn.batch_normalization(branch6_2,offset=beta_38,\
                                              mean=moving_mean_38, \
                                              variance=moving_variance_38,scale=None,variance_epsilon=0.001)
        #branch 3
        branch6_3 = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_39 = self.var_dict['InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/weights']
        branch6_3 = tf.nn.conv2d(branch6_3, w_39, strides=[1, 1, 1, 1], padding='SAME')
        branch6_3 = tf.nn.relu(branch6_3)
        beta_39 = self.var_dict['InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/beta']
        moving_mean_39 = self.var_dict['InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean']
        moving_variance_39 = self.var_dict['InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance']
        branch6_3 = tf.nn.batch_normalization(branch6_3,offset=beta_39,\
                                              mean=moving_mean_39, \
                                              variance=moving_variance_39,scale=None,variance_epsilon=0.001)
        incpt = tf.concat(
            axis=3, values=[branch6_0, branch6_1, branch6_2, branch6_3])
        #seventh inception
        #branch 0
        w_40 = self.var_dict['InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/weights']
        branch7_0 = tf.nn.conv2d(incpt, w_40, strides=[1, 1, 1, 1], padding='SAME')
        branch7_0 = tf.nn.relu(branch7_0)
        beta_40 = self.var_dict['InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_40 = self.var_dict['InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_40 = self.var_dict['InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch7_0 = tf.nn.batch_normalization(branch7_0,offset=beta_40,\
                                              mean=moving_mean_40, \
                                              variance=moving_variance_40,scale=None,variance_epsilon=0.001)
        #branch 1
        w_41 = self.var_dict['InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/weights']
        branch7_1 = tf.nn.conv2d(incpt, w_41, strides=[1, 1, 1, 1], padding='SAME')
        branch7_1 = tf.nn.relu(branch7_1)
        beta_41 = self.var_dict['InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_41 = self.var_dict['InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_41 = self.var_dict['InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch7_1 = tf.nn.batch_normalization(branch7_1,offset=beta_41,\
                                              mean=moving_mean_41, \
                                              variance=moving_variance_41,scale=None,variance_epsilon=0.001)
        w_42 = self.var_dict['InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/weights']
        branch7_1 = tf.nn.conv2d(branch7_1, w_42, strides=[1, 1, 1, 1], padding='SAME')
        branch7_1 = tf.nn.relu(branch7_1)
        beta_42 = self.var_dict['InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/beta']
        moving_mean_42 = self.var_dict['InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean']
        moving_variance_42 = self.var_dict['InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance']
        branch7_1 = tf.nn.batch_normalization(branch7_1,offset=beta_42,\
                                              mean=moving_mean_42, \
                                              variance=moving_variance_42,scale=None,variance_epsilon=0.001)
        #branch 2
        w_43 = self.var_dict['InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/weights']
        branch7_2 = tf.nn.conv2d(incpt, w_43, strides=[1, 1, 1, 1], padding='SAME')
        branch7_2 = tf.nn.relu(branch7_2)
        beta_43 = self.var_dict['InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_43 = self.var_dict['InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_43 = self.var_dict['InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch7_2 = tf.nn.batch_normalization(branch7_2,offset=beta_43,\
                                              mean=moving_mean_43, \
                                              variance=moving_variance_43,scale=None,variance_epsilon=0.001)
        w_44 = self.var_dict['InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/weights']
        branch7_2 = tf.nn.conv2d(branch7_2, w_44, strides=[1, 1, 1, 1], padding='SAME')
        branch7_2 = tf.nn.relu(branch7_2)
        beta_44 = self.var_dict['InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/beta']
        moving_mean_44 = self.var_dict['InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean']
        moving_variance_44 = self.var_dict['InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance']
        branch7_2 = tf.nn.batch_normalization(branch7_2,offset=beta_44,\
                                              mean=moving_mean_44, \
                                              variance=moving_variance_44,scale=None,variance_epsilon=0.001)
        #branch 3
        branch7_3 = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_45 = self.var_dict['InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/weights']
        branch7_3 = tf.nn.conv2d(branch7_3, w_45, strides=[1, 1, 1, 1], padding='SAME')
        branch7_3 = tf.nn.relu(branch7_3)
        beta_45 = self.var_dict['InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/beta']
        moving_mean_45 = self.var_dict['InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean']
        moving_variance_45 = self.var_dict['InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance']
        branch7_3 = tf.nn.batch_normalization(branch7_3,offset=beta_45,\
                                              mean=moving_mean_45, \
                                              variance=moving_variance_45,scale=None,variance_epsilon=0.001)
        incpt = tf.concat(
            axis=3, values=[branch7_0, branch7_1, branch7_2, branch7_3])
        incpt = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                               strides=[1, 2, 2, 1], padding='SAME')
        #eighth inception
        #branch 0
        w_46 = self.var_dict['InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/weights']
        branch8_0 = tf.nn.conv2d(incpt, w_46, strides=[1, 1, 1, 1], padding='SAME')
        branch8_0 = tf.nn.relu(branch8_0)
        beta_46 = self.var_dict['InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_46 = self.var_dict['InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_46 = self.var_dict['InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch8_0 = tf.nn.batch_normalization(branch8_0,offset=beta_46,\
                                              mean=moving_mean_46, \
                                              variance=moving_variance_46,scale=None,variance_epsilon=0.001)
        #branch 1
        w_47 = self.var_dict['InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/weights']
        branch8_1 = tf.nn.conv2d(incpt, w_47, strides=[1, 1, 1, 1], padding='SAME')
        branch8_1 = tf.nn.relu(branch8_1)
        beta_47 = self.var_dict['InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_47 = self.var_dict['InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_47 = self.var_dict['InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch8_1 = tf.nn.batch_normalization(branch8_1,offset=beta_47,\
                                              mean=moving_mean_47, \
                                              variance=moving_variance_47,scale=None,variance_epsilon=0.001)
        w_48 = self.var_dict['InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/weights']
        branch8_1 = tf.nn.conv2d(branch8_1, w_48, strides=[1, 1, 1, 1], padding='SAME')
        branch8_1 = tf.nn.relu(branch8_1)
        beta_48 = self.var_dict['InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/beta']
        moving_mean_48 = self.var_dict['InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean']
        moving_variance_48 = self.var_dict['InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance']
        branch8_1 = tf.nn.batch_normalization(branch8_1,offset=beta_48,\
                                              mean=moving_mean_48, \
                                              variance=moving_variance_48,scale=None,variance_epsilon=0.001)
        #branch 2
        w_49 = self.var_dict['InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/weights']
        branch8_2 = tf.nn.conv2d(incpt, w_49, strides=[1, 1, 1, 1], padding='SAME')
        branch8_2 = tf.nn.relu(branch8_2)
        beta_49 = self.var_dict['InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_49 = self.var_dict['InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_49 = self.var_dict['InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch8_2 = tf.nn.batch_normalization(branch8_2,offset=beta_49,\
                                              mean=moving_mean_49, \
                                              variance=moving_variance_49,scale=None,variance_epsilon=0.001)
        w_50 = self.var_dict['InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/weights']
        branch8_2 = tf.nn.conv2d(branch8_2, w_50, strides=[1, 1, 1, 1], padding='SAME')
        branch8_2 = tf.nn.relu(branch8_2)
        beta_50 = self.var_dict['InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/beta']
        moving_mean_50 = self.var_dict['InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/moving_mean']
        moving_variance_50 = self.var_dict['InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/BatchNorm/moving_variance']
        branch8_2 = tf.nn.batch_normalization(branch8_2,offset=beta_50,\
                                              mean=moving_mean_50, \
                                              variance=moving_variance_50,scale=None,variance_epsilon=0.001)
        #branch 3
        branch8_3 = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_51 = self.var_dict['InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/weights']
        branch8_3 = tf.nn.conv2d(branch8_3, w_51, strides=[1, 1, 1, 1], padding='SAME')
        branch8_3 = tf.nn.relu(branch8_3)
        beta_51 = self.var_dict['InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta']
        moving_mean_51 = self.var_dict['InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean']
        moving_variance_51 = self.var_dict['InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance']
        branch8_3 = tf.nn.batch_normalization(branch8_3,offset=beta_51,\
                                              mean=moving_mean_51, \
                                              variance=moving_variance_51,scale=None,variance_epsilon=0.001)
        incpt = tf.concat(
            axis=3, values=[branch8_0, branch8_1, branch8_2, branch8_3])
        #ninth inception
        #branch 0
        w_52 = self.var_dict['InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/weights']
        branch9_0 = tf.nn.conv2d(incpt, w_52, strides=[1, 1, 1, 1], padding='SAME')
        branch9_0 = tf.nn.relu(branch9_0)
        beta_52 = self.var_dict['InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_52 = self.var_dict['InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_52 = self.var_dict['InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch9_0 = tf.nn.batch_normalization(branch9_0,offset=beta_52,\
                                              mean=moving_mean_52, \
                                              variance=moving_variance_52,scale=None,variance_epsilon=0.001)
        #branch 1
        w_53 = self.var_dict['InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/weights']
        branch9_1 = tf.nn.conv2d(incpt, w_53, strides=[1, 1, 1, 1], padding='SAME')
        branch9_1 = tf.nn.relu(branch9_1)
        beta_53 = self.var_dict['InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_53 = self.var_dict['InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_53 = self.var_dict['InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch9_1 = tf.nn.batch_normalization(branch9_1,offset=beta_53,\
                                              mean=moving_mean_53, \
                                              variance=moving_variance_53,scale=None,variance_epsilon=0.001)
        w_54 = self.var_dict['InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/weights']
        branch9_1 = tf.nn.conv2d(branch9_1, w_54, strides=[1, 1, 1, 1], padding='SAME')
        branch9_1 = tf.nn.relu(branch9_1)
        beta_54 = self.var_dict['InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/beta']
        moving_mean_54 = self.var_dict['InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean']
        moving_variance_54 = self.var_dict['InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance']
        branch9_1 = tf.nn.batch_normalization(branch9_1,offset=beta_54,\
                                              mean=moving_mean_54, \
                                              variance=moving_variance_54,scale=None,variance_epsilon=0.001)
        #branch 2
        w_55 = self.var_dict['InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/weights']
        branch9_2 = tf.nn.conv2d(incpt, w_55, strides=[1, 1, 1, 1], padding='SAME')
        branch9_2 = tf.nn.relu(branch9_2)
        beta_55 = self.var_dict['InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta']
        moving_mean_55 = self.var_dict['InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean']
        moving_variance_55 = self.var_dict['InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance']
        branch9_2 = tf.nn.batch_normalization(branch9_2,offset=beta_55,\
                                              mean=moving_mean_55, \
                                              variance=moving_variance_55,scale=None,variance_epsilon=0.001)
        w_56 = self.var_dict['InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/weights']
        branch9_2 = tf.nn.conv2d(branch9_2, w_56, strides=[1, 1, 1, 1], padding='SAME')
        branch9_2 = tf.nn.relu(branch9_2)
        beta_56 = self.var_dict['InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta']
        moving_mean_56 = self.var_dict['InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean']
        moving_variance_56 = self.var_dict['InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance']
        branch9_2 = tf.nn.batch_normalization(branch9_2,offset=beta_56,\
                                              mean=moving_mean_56, \
                                              variance=moving_variance_56,scale=None,variance_epsilon=0.001)
        #branch 3
        branch9_3 = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1], \
                                 strides=[1, 1, 1, 1], padding='SAME')
        w_57 = self.var_dict['InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/weights']
        branch9_3 = tf.nn.conv2d(branch9_3, w_57, strides=[1, 1, 1, 1], padding='SAME')
        branch9_3 = tf.nn.relu(branch9_3)
        beta_57 = self.var_dict['InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta']
        moving_mean_57 = self.var_dict['InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean']
        moving_variance_57 = self.var_dict['InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance']
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
        return nets,test

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
        # tf.summary.scalar('random_negative_loss', rand_neg)
        # tf.summary.scalar('positive_loss', pos)
        tf.summary.scalar('total_loss', loss)

        return loss, eucd_p, eucd_n1, eucd_n2

    def create_indices(self,labels):
        old = labels[0]
        indices = dict()
        indices[old] = 0
        for x in xrange(len(labels) - 1):
            new = labels[x + 1]
            if old != new:
                indices[new] = x + 1
            old = new
        return indices

    def generate_triplet(self,_labels, _n_samples):
        # retrieve loaded patches and labels
        labels = _labels
        # group labels in order to have O(1) search
        count = collections.Counter(labels)
        # index the labels in order to have O(1) search
        indices = self.create_indices(labels)
        # range for the sampling
        labels_size = len(labels) - 1
        # triplets ids
        _index_1 = []
        _index_2 = []
        _index_3 = []
        # generate the triplets
        pbar = xrange(_n_samples)

        for x in pbar:
            idx = random.randint(0, labels_size)
            num_samples = count[labels[idx]]
            begin_positives = indices[labels[idx]]

            offset_a, offset_p = random.sample(xrange(num_samples), 2)
            while offset_a == offset_p:
                offset_a, offset_p = random.sample(xrange(num_samples), 2)
            idx_a = begin_positives + offset_a
            idx_p = begin_positives + offset_p
            _index_1.append(idx_a)
            _index_2.append(idx_p)
            idx_n = random.randint(0, labels_size)
            while labels[idx_n] == labels[idx_a] and \
                            labels[idx_n] == labels[idx_p]:
                idx_n = random.randint(0, labels_size)
            _index_3.append(idx_n)

        _index_1 = np.array(_index_1)
        _index_2 = np.array(_index_2)
        _index_3 = np.array(_index_3)

        temp_index = np.arange(_index_1.shape[0])

        np.random.shuffle(temp_index)
        _index_1 = _index_1[temp_index]
        _index_2 = _index_2[temp_index]
        _index_3 = _index_3[temp_index]

        return _index_1, _index_2, _index_3

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    cnn_triplet = CNN_Triplet_Metric(sess=sess)


