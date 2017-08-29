import tensorflow as tf


def triplet_loss(margins,oa,op,on):
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