import tensorflow as tf
import scipy.io
import numpy as np


#this ./ is for version 1.0 and later
# ckpt_path = './inception_v1.ckpt'

# # tensors = tf.contrib.framework.list_variables(ckpt_path)
# #
# # for ts in tensors:
# #     print(ts)


# # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# #
# # # List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
# # print_tensors_in_checkpoint_file(file_name='./inception_v1.ckpt', tensor_name='',all_tensors=True)




# import numpy as np

# var1 = tf.get_variable(shape=[7,7,3,64], name='InceptionV1/Conv2d_1a_7x7/weights')



# with tf.Session() as sess:
#     saver = tf.train.Saver(var_list=[var1])
#     saver.restore(sess,"./inception_v1.ckpt")

#     w = var1.eval()
#     w = np.transpose(w,(3,2,1,0))
#     print w

pretrained_weights = scipy.io.loadmat('tf_ckpt_from_caffe.mat')
print pretrained_weights['conv1/7x7_s2'][1,1,1,2]
tran = np.transpose(pretrained_weights['conv1/7x7_s2'],(3,2,1,0))

print tran[1,2,1,1]