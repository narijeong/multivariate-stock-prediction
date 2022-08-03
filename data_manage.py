import numpy as np
import pandas as pd
import os

# create prepocessed data
def preprocess(data_path, fname):
    df = pd.read_csv(data_path + fname)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    df['Close-1'] = df['Close'].shift(1)
    try: 
        df['c_open'] = (df['Original_Open']/df['Close']) - 1
    except:
        df['c_open'] = (df['Open']/df['Close']) - 1
    df['c_high'] = df['High']/df['Close'] - 1
    df['c_low'] = df['Low']/df['Close'] - 1
    df['c_close'] = df['Close']/df['Close-1'] - 1

    df['adj_close-1'] = df['Adj Close'].shift(1)
    df['c_adj_close'] = df['Adj Close']/df['adj_close-1'] -1

    df['adj_close+1'] = df['Adj Close'].shift(-1)
    df['c_adj_close+1'] = df['adj_close+1']/df['Adj Close'] -1

    df['c_5'] = (df['Adj Close'].rolling(5).sum()/(5*df['Adj Close'])) - 1
    df['c_10'] = (df['Adj Close'].rolling(10).sum()/(10*df['Adj Close'])) - 1
    df['c_15'] = (df['Adj Close'].rolling(15).sum()/(15*df['Adj Close'])) - 1
    df['c_20'] = (df['Adj Close'].rolling(20).sum()/(20*df['Adj Close'])) - 1
    df['c_25'] = (df['Adj Close'].rolling(25).sum()/(25*df['Adj Close'])) - 1
    df['c_30'] = (df['Adj Close'].rolling(30).sum()/(30*df['Adj Close'])) - 1
    df['c_label'] = 0
    # df.loc[df['c_adj_close'] > 0.0055, 'c_label'] = 1
    # df.loc[df['c_adj_close'] < -0.005, 'c_label'] = -1

    # df.loc[df['c_adj_close+1'] > 0.0055, 'c_label'] = 1
    # df.loc[df['c_adj_close+1'] < -0.005, 'c_label'] = -1
    df.loc[df['c_adj_close+1'] >= 0, 'c_label'] = 1
    df.loc[df['c_adj_close+1'] < 0, 'c_label'] = -1
    df = df[['c_open', 'c_high', 'c_low', 'c_close','c_adj_close','c_5', 'c_10', 'c_15', 'c_20', 'c_25','c_30', 'c_label', 'Adj Close']]
    # df = df.iloc[:-1,:]
    df.iloc[:29, :] = -123321.000000
    return df