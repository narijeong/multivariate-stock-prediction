from multiprocessing.dummy import active_children
import os
from re import I
import sys
import logging
import argparse
import json
import pandas as pd
import numpy as np

from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import Lambda, Concatenate, Reshape
from keras.layers import MultiHeadAttention
from keras.models import Model 

from keras.utils.vis_utils import plot_model

from myattention import Attention
from layers import Aggregate
from load import load_data, load_cla_data

def train(model):
    # tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt = load_data(args.path + "/index", "SPY_processed.csv",
    #         tra_date, val_date, tes_date, seq=args.time_steps)

    # tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt = load_cla_data(args.path,
    #     tra_date, val_date, tes_date, seq=args.time_steps)
    x1 = np.array([[[1,2,3,4,5,6,7,8,9,10,11],
                   [1,2,3,4,5,6,7,8,9,10,11]],
                   [[1,2,3,4,5,6,7,8,9,10,11],
                   [1,2,3,4,5,6,7,8,9,10,11]],])
    x2 = np.array([[[1,2,3,4,5,6,7,8,9,10,11],
                   [1,2,3,4,5,6,7,8,9,10,11]],
                   [[1,2,3,4,5,6,7,8,9,10,11],
                   [1,2,3,4,5,6,7,8,9,10,11]],])
    y = np.array([3,4])
    history = model.fit(x=[x1,x2], y=y, epochs=5)
    print(history.history)

    # history = model.fit(x =[index_data['X_train'],stock_data[i]['X_train']], y=stock_data[i]['y_train'],
    #                     validation_data=([index_data['X_val'],stock_data['X_val']], stock_data[i]['y_val']),
    #                     epochs=args.epochs
    #                     )

def transform_features(features, ticker_size):
    transformed  = np.zeros(params['sample_size'], ticker_size, params['feature_dim'],
                              dtype=np.float32)
    # print(features)
    # context_vector_size = features.shape[1]
    # sampe_size = params['sampe_size']
    # print('sample_size', params['sampe_size'])
    for i in range(params['sample_size']):
        stock_context_vector = []
        for j in range(ticker_size, ticker_size):
            temp = features[i+j]
            stock_context_vector.append(temp)
        transformed = np.add(transformed, stock_context_vector)
    return transformed

def construct_model():
    index_input = Input(shape=(args.time_steps, params['feature_dim']))
    stock_input = Input(shape=(args.time_steps, params['feature_dim']))

    index_model = time_axis_attention(index_input, params['feature_dim'], args.units)

    stock_models = [None]*params['ticker_size']
    context_aggs= [None]*params['ticker_size']
    context_agg_models = [None]*params['ticker_size']
    
    for i in range(params['ticker_size']):
        stock_models[i] = time_axis_attention(stock_input, params['feature_dim'], args.units)
        context_aggs[i] = Aggregate(32)([index_model, stock_models[i]])
        # context_agg_models[i] = Model(inputs=[index_input, stock_input], outputs=context_aggs[i])

    last_context_agg_model = Model(inputs=[index_input, stock_input], outputs=context_aggs[i])
    plot_model(last_context_agg_model, to_file='contex_agg.png', show_shapes=True, show_layer_names=True)
    
    c_model = context_aggs[0]
    for i in range(1, len(context_agg_models)):
        c_model = Concatenate()([c_model, context_aggs[i]])

    # Data Transformation to H (d*n)
    assert(c_model.shape[1] == params['ticker_size']*args.units)
    print(c_model.shape)
    # data axis context
    c_model = Reshape((params['ticker_size'], args.units))(c_model)
    print(c_model.shape)

    # Data Axis context
    mha = MultiHeadAttention(num_heads=2, key_dim=2)(c_model)
    #Nonlinear Transformation.


    model = mha
    # model.compile(loss='mae', optimizer='adam')
    plot_model(model, to_file='model_plot4.png', show_shapes=True, show_layer_names=True)

    return model

def time_axis_attention(model_input, feature_dim, units):
    x = Dense(feature_dim, activation='tanh')(model_input)
    x = LSTM(units, return_sequences=True)(x)
    x = Attention(feature_dim)(x)
    x = BatchNormalization()(x)
    return x

if __name__ == '__main__':
    desc = 'the DTML model'
    parser = argparse.ArgumentParser(description=desc)
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

    tra_date = '2007-02-14'
    val_date = '2015-01-02'
    tes_date = '2016-01-04'
    
    params = {
        "feature_dim": 11,
        "ticker_size": 50,
        "batch_size": 2,
        "sample_size": 1980,
    }
    construct_model()
    # train(model)