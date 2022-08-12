import nni
import argparse
from load_data import load_cla_data
from keras import Model, layers, backend, activations
from keras.utils.vis_utils import plot_model
from attention import Attention
from layers import *
import pandas as pd
import numpy as np

def run_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path of pv data', type=str,
                        default='./data/kdd17/ourpped')
    parser.add_argument('--batch_size', help='batch size', type=str,
                        default=128)
    parser.add_argument('--seq', help='length of history', type=int,
                        default=5)
    parser.add_argument('--epochs', help='epochs for training', type=int,
                        default=1)
    args = parser.parse_args()
    
    return args


def build_model(args, params):
    def input_reshape(x):
            return backend.reshape(x, (-1, args['seq'], params['fea_dim']))

    def backend_reshape(x):
        return backend.reshape(x, (-1, params['ticker_num'] + 1, params['fea_dim']))

    input_shape = layers.Input(shape=(params['ticker_num'] + 1, args['seq'], params['fea_dim']), name='index_input')
    print(input_shape)
    x = layers.Lambda(input_reshape)(input_shape)
    x = layers.Dense(params['fea_dim'], activation='tanh')(x)
    
    print('dtml')
    x = layers.LSTM(params['fea_dim'], return_sequences=True)(x)
    x = Attention(params['fea_dim'])(x)
    print('attention shape', x.shape)
    x = layers.LayerNormalization()(x)
    x = layers.Lambda(backend_reshape)(x)
    print('backend_reshape shape', x.shape)
    x = layers.Lambda(lambda x: x + x[0,0])(x)
    # x = ContextAggreation(x)
    x = layers.Lambda(lambda x: x[:, 1:])(x)
    print('layer shape', x.shape)

    # Data Axis contextx
    mha = layers.MultiHeadAttention(num_heads=8, key_dim=2)(x,x)
    # Nonlinear Transformation
    add = layers.Add()([x, mha])
    x = layers.Dense(add.shape[-1])(add)
    x = layers.Add()([add, x])
    x = layers.Dropout(rate=0.2)(x)
    x = activations.tanh(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    print('final shape', x.shape)
    model = Model(inputs=input_shape, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    plot_model(model, to_file='./images/model.png', show_shapes=True, show_layer_names=True)
    return model

def main(args):
    tra_date = '2007-01-03'
    val_date = '2015-01-02'
    tes_date = '2016-01-04'

    # load data
    tra_pv, tra_wd, tra_gt, \
    val_pv, val_wd, val_gt, \
    tes_pv, tes_wd, tes_gt = load_cla_data(args['path'], tra_date, val_date, tes_date, seq=args['seq'])
    params = {'fea_dim':  tra_pv.shape[-1], 
              'ticker_num': tra_pv.shape[1] - 1
             }
    
    # build model
    model = build_model(args, params)
    
    # train
    history = model.fit(x=tra_pv, y=tra_gt, validation_data=(val_pv, val_gt), epochs=args['epochs'])
    pred_gt = model.predict(tes_pv)
    pd.DataFrame(np.squeeze(pred_gt)).to_csv("./output/prediction.csv")
    _, acc = model.evaluate(tes_pv, tes_gt, verbose=0)
    print('Final result is: %d', acc)
    nni.report_final_result(acc)

if __name__ == '__main__':
    params = nni.get_next_parameter() #It is a dictionary object.
    print('get params')

    args_dict = vars(run_args()) #get dict
    args_dict.update(params)
    print(params)
    args = argparse.Namespace(**args_dict)

    main(params)