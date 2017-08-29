import tensorflow as tf


#this ./ is for version 1.0 and later
ckpt_path = './inception_v1.ckpt'

tensors = tf.contrib.framework.list_variables(ckpt_path)

for ts in tensors:
    print(ts)