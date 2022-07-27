from keras.layers import Layer
from keras.layers import add
import tensorflow as tf

class Aggregate(Layer):

    def __init__(self, units=32, name="aggregate"):
        super(Aggregate, self).__init__(name=name)
        self.units = units

    def build(self, input_shape):  # Create the state of the layer (weights)
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
        initial_value=w_init(shape=(input_shape[0][-1], self.units),
                             dtype='float32'),
                            trainable=True)

    def call(self, inputs):  # Defines the computation from inputs to outputs
        a = inputs[0]
        b = inputs[1]
        return add([tf.matmul(a, self.w), b])

