import argparse
from keras.layers import Dense
from keras.layers import LSTM

def time_axis_attention(data_path, fname):
    print(fname)
    tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt = load_data(data_path, fname,
            tra_date, val_date, tes_date, seq=params['time_steps']
        ) 
    model_input = Input(shape=(params['time_steps'], params['input_dim']))
    x = Dense(params['input_dim'])(model_input)
    x = LSTM(64, return_sequences=True)(x)
    x = Attention(params['input_dim'])(x)
    model = Model(model_input, x)
    model.compile(loss='mae', optimizer='adam')
    # context vector
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-1].output)
    intermediate_output = intermediate_layer_model(tra_pv).numpy()
    return intermediate_output



# # aggregate market context with individual stock context
# # summary the indivisual context with a global trend
# def context_aggregation():
#     pass
# # learn the stock correlation by a transformer
# def data_axis_attention():
#     pass



# if __name__ == '__main__':
#     print('main.py')
#     load_data()
#     time_axis_attention()
#     context_aggregation()
#     data_axis_attention()

if __name__ == '__main__':
    desc = 'the DTML model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', '--path', help='path of pv data', type=str,
                        default='./data/stocknet-dataset/price/ourpped')
    parser.add_argument('-l', '--seq', help='length of history', type=int,
                        default=5)
    parser.add_argument('-u', '--unit', help='number of hidden units in lstm',
                        type=int, default=32)
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
    parser.add_argument('-e', '--epoch', help='epoch', type=int, default=150)
    # parser.add_argument('-r', '--learning_rate', help='learning rate',
    #                     type=float, default=1e-2)