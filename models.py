
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import BatchNormalization

from attention import Attention


def attention_lstm(model_input):
    x = Dense(params['input_dim'])(model_input)
    x = LSTM(64, return_sequences=True)(x)
    x = Attention(params['input_dim'])(x)
    x = BatchNormalization()(x)
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