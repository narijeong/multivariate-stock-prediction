from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def load_data();
    pass

def feature_mapping():
    pass

# summarize the historical prices of each stock
def time_axis_attention():
    features_size = 11
    output_size = 1
    model = Sequential()
    model.add(LSTM((output_size), batch_input_shape=(None, features_size, output_size), return_sequences=True))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

# aggregate market context with individual stock context
# summary the indivisual context with a global trend
def context_aggregation():
    pass
# learn the stock correlation by a transformer
def data_axis_attention():
    pass



if __name__ == '__main__':
    print('main.py')
    load_data()
    time_axis_attention()
    context_aggregation()
    data_axis_attention()
