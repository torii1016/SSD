import tensorflow as tf

def smooth_L1(x):
    sml1 = tf.multiply( 0.5, tf.pow(x, 2.0) )
    sml2 = tf.subtract( tf.abs(x), 0.5 )
    cond = tf.less( tf.abs(x), 1.0 )
    return tf.where( cond, sml1, sml2 )