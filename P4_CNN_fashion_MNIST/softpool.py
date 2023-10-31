# From https://github.com/qwopqwop200/SoftPool/blob/main/tensorflow_softpool.py#L14-#L22 

import tensorflow as tf

class SoftPooling2D(tf.keras.layers.Layer):
    def __init__(self,pool_size=(2, 2),strides=None,padding='valid',data_format=None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = tf.keras.layers.AvgPool2D(pool_size,strides,padding,data_format)
    def call(self, x):
        x_exp = tf.math.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool