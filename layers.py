from keras.layers import Layer
from keras.layers import add
import tensorflow as tf
import pandas as pd

class Aggregate(Layer):

    def __init__(self, units=32):
        super(Aggregate, self).__init__()
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


class ContextAggreation(Layer):
    def __init__(self, units=32):
        super(ContextAggreation, self).__init__()
        self.units = units

    def build(self, input_shape):  # Create the state of the layer (weights)
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
        initial_value=w_init(shape=(input_shape[-1], self.units),
                             dtype='float32'),
                            trainable=True)

    def call(self, inputs):  # Defines the computation from inputs to outputs
        function_to_map = lambda x: x + x[0]
        result = tf.map_fn(function_to_map, inputs)

        print('result', result.shape)
        return result

# import numpy as np

# x = np.array([[[1,2,3,4,5,6,7,8,9,10,11],
#                    [2,3,4,5,6,7,8,9,10,11,12],
#                    [3,4,5,6,7,8,9,10,11,12,13]],
#                    [[1,2,3,4,5,6,7,8,9,10,11],
#                    [2,3,4,5,6,7,8,9,10,11,12],
#                    [3,4,5,6,7,8,9,10,11,12,13]],
#                    [[1,2,3,4,5,6,7,8,9,10,11],
#                    [2,3,4,5,6,7,8,9,10,11,12],
#                    [3,4,5,6,7,8,9,10,11,12,13]],
#                    [[1,2,3,4,5,6,7,8,9,10,11],
#                    [2,3,4,5,6,7,8,9,10,11,12],
#                    [3,4,5,6,7,8,9,10,11,12,13]],
#                    [[1,2,3,4,5,6,7,8,9,10,11],
#                    [2,3,4,5,6,7,8,9,10,11,12],
#                    [3,4,5,6,7,8,9,10,11,12,13]],])

# # model = tf.keras.Sequential()
# # model.add(tf.keras.Input(shape=(3,2)))
# # model.add(tf.keras.layers.Dense(1))
# # print()
# # print(x.shape)

# # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # model.fit(x, [])


# x = tf.keras.Input(shape=(3,11))
# ContextAggreation()(x)
