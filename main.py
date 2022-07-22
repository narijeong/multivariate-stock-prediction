import os
import sys
import logging
import argparse
import json
import pandas as pd

from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.models import Model
from keras.layers import MultiHeadAttention
from keras.utils.vis_utils import plot_model

from attention import Attention
from keract import get_activations
from aggregate import *
from load import load_data

def prepare_data():
    index_data =  {}
    tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt = load_data(args.path + "/index", "SPY_processed.csv",
            tra_date, val_date, tes_date, seq=args.time_steps)
    index_data = {
            "X_train": tra_pv,
            "y_train": tra_gt,
            "X_val": val_pv,
            "y_val": val_gt,
            "X_test": tes_pv,
            "y_test": tes_gt,
            }

    fnames = [fname for fname in os.listdir(args.path + '/processed') if
            os.path.isfile(os.path.join(args.path + '/processed', fname))]
    stcok_data = [{}]*len(fnames)
    for idx, fname in enumerate(fnames):
        tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt = load_data(args.path + "/processed", fname,
        tra_date, val_date, tes_date, seq=args.time_steps)
        stcok_data[idx] = {
                        "X_train": tra_pv,
                        "y_train": tra_gt,
                        "X_val": val_pv,
                        "y_val": val_gt,
                        "X_test": tes_pv,
                        "y_test": tes_gt,
                        }
    return index_data, stcok_data

def time_axis_attention(model_input, feature_dim, units):
    x = Dense(feature_dim)(model_input)
    x = LSTM(units, return_sequences=True)(x)
    x = Attention(feature_dim)(x)
    x = BatchNormalization()(x)
    return x

def context_aggregation(index_data, stock_data):
    stock_model = [None]*len(stock_data)
    model = [None]*len(stock_data)
    
    index_input = Input(shape=(args.time_steps, g_params['feature_dim']))
    stock_input = Input(shape=(args.time_steps, g_params['feature_dim']))

    context_aggregation_result = []

    index_model = time_axis_attention(index_input, g_params['feature_dim'], args.units)
    index_model = Model(inputs=index_input, outputs=index_model)

    for i, stock in enumerate(stock_data[:3]):
        stock_model[i] = time_axis_attention(stock_input, g_params['feature_dim'], args.units)
        stock_model[i] = Model(inputs=stock_input, outputs=stock_model[i])
        merged = Aggregate(g_params['feature_dim'])([index_model.output, stock_model[i].output])
        # # z = Dense(1, activation="sigmoid")(merged)
        model[i] = Model(inputs=[index_model.input, stock_model[i].input], outputs=merged)
        model[i].compile(loss='mae', optimizer='adam')
        plot_model(model[i], to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        history = model[i].fit(x =[index_data['X_train'],stock_data[i]['X_train']], y=stock_data[i]['y_train'],
                            validation_data=([index_data['X_val'],stock_data[i]['X_val']], stock_data[i]['y_val']),
                            epochs=args.epochs
                            )
        # print(history.history)
        context_aggregation_result.append(model[i].predict([index_data['X_train'],stock_data[i]['X_train']]))

    return context_aggregation_result
    
def reshape_data(input):
    print(len(input))
    print(len(input[0]))
    # result = [[None] for x in range(len(input[0][1]))]
    # print(len(input[0][0]))
    # print(len(input[0][1]))
    # for i in range(len(input[0][1])): #1980
    #     for j in range(len(input[0][0])): #50
        
    #         result[i][j] = input[j][i]

    # # print(result.shape)
    # return result 

def data_axis_attention(context_aggregation_result):
    print(context_aggregation_result[19])
    print(context_aggregation_result[19].shape)
    
    model_input = Input(shape=(50,11))
    x = MultiHeadAttention(num_heads=2, key_dim=2)(model_input)
    # x = onlineartransformation()
    x = Dense(1, activation='sigmoid')(x)
    model = Model(input=model_input, output = x)
    return model
    
    

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
    
    index_data, stock_data = prepare_data()
    g_params = {
        "feature_dim": index_data["X_train"].shape[2],
    }
    # print(index_data['X_train'].shape)
    # print(g_params['feature_dim'])
    # print(stock_data[0]['X_train'].shape)
    context_aggregation_result = context_aggregation(index_data, stock_data)
    context_aggregation_result = reshape_data(context_aggregation_result)
    # data_axis_attention(context_aggregation_result)
    # print(index_data['X_test'].shape)
    # print(stock_data[0]['X_test'].shape)
    # print(stock_data[0]['y_train'])