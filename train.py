from multiprocessing.dummy import active_children
import os
from re import I
import sys
import logging
import argparse

import json
import pandas as pd
import numpy as np

from keras import activations
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import Lambda, Concatenate, Reshape, Add, Dropout
from keras.layers import MultiHeadAttention
from keras.models import Model 

from keras.utils.vis_utils import plot_model

from myattention import Attention
from layers import Aggregate
from load import load_data, load_cla_data

# market to choose from
# kospi200
# snp500

class DTML:
    def __init__(self, index, data_path, time_steps,):
        self.index = index
        self.time_steps = time_steps
        self.data_path = data_path
        self.params = {}
        self.feature_dim = 0
        self.sample_size = 0
        self.ticker_size = 0
        self.prepare_data()


    # prepare tarin, validation, test data
    def prepare_data(self):
        index_data = pd.read_csv('./data/index/preprocessed/' + self.index +'_p.csv')

        


    def create_model(self):
        def time_axis_attention(model_input, feature_dim, units):
            x = Dense(feature_dim, activation='tanh')(model_input)
            x = LSTM(units, return_sequences=True)(x)
            x = Attention(feature_dim)(x)
            x = BatchNormalization()(x)
            return x

        index_input = Input(shape=(self.time_steps, self.params['feature_dim']), name='index_input')
        index_model = time_axis_attention(index_input, self.params['feature_dim'], self.units)
        # index_model = Model(inputs=index_input, outputs=index_model)

        stock_input_list = [Input(shape=(self.time_steps, self.params['feature_dim'])) for stock_input in range(self.params['ticker_size'])]

        context_aggs = [None]*(self.params['ticker_size'])
        for i in range(self.params['ticker_size']):
            stock_input = stock_input_list[i]
            stock_model = time_axis_attention(stock_input, self.params['feature_dim'], self.units)
            # stock_model = Model(inputs=stock_input, outputs=stock_model)
            context_aggs[i] = Aggregate(self.units)([index_model, stock_model])

        x = context_aggs[0]
        for i in range(1, self.params['ticker_size']):
            x = Concatenate()([x, context_aggs[i]])
        x = Reshape((self.params['ticker_size'], -1))(x)
        print(x.shape)

        # Data Axis context
        mha = MultiHeadAttention(num_heads=2, key_dim=2)(x, x)
        # Nonlinear Transformation
        addition = Add()([x, mha])
        x = Dense(self.units)(addition)
        x = Add()([addition, x])
        x = Dropout(rate=0.2)(x)
        x = activations.tanh(x)
        x = Dense(1, activation='sigmoid')(x)
        print(x.shape)
        model = Model(inputs=[index_input, *stock_input_list], outputs=x)
        model.compile(loss='mae', optimizer='adam')
        plot_model(model, to_file='./image/model_plot.png', show_shapes=True, show_layer_names=True)
        return model


    def train(self):
        model = self.create_model()





if __name__ == '__main__':
    desc = 'The DTML model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--index', help='market to use as trend', type=str,
                        default='snp500')
    parser.add_argument('-p', '--path', help='path of data', type=str,
                        default='./data/kdd17')
    parser.add_argument('-t', '--time_steps', help='length of history', type=int,
                        default=5)
    parser.add_argument('-u', '--units', help='number of hidden units in lstm',
                        type=int, default=32)
    parser.add_argument('-e', '--epochs', help='epochs', type=int, default=5)
    # parser.add_argument('-r', '--learning_rate', help='learning rate',
    #                     type=float, default=1e-2)
    # parser.add_argument('-l2', '--alpha_l2', type=float, default=1e-2,
    #                     help='alpha for l2 regularizer')
    # parser.add_argument('-la', '--beta_adv', type=float, default=1e-2,
    #                     help='beta for adverarial loss')
    # parser.add_argument('-le', '--epsilon_adv', type=float, default=1e-2,
    #                     help='epsilon to control the scale of noise')
    # parser.add_argument('-s', '--step', help='steps to make prediction',
    #                     type=int, default=1)
    # parser.add_argument('-b', '--batch_size', help='batch size', type=int,
    #                     default=1024)

    args = parser.parse_args()

    dtml = DTML(args.data_path, args.time_step)

