from datetime import datetime
import numpy as np
import os
import pandas as pd

# split single data file into train, validation and test
def load_data(data_path, fname, feature_dim, time_steps=5):
    print(fname)

    # data = np.genfromtxt(
    #     os.path.join(data_path, fname), dtype=float, delimiter=',',
    #     skip_header=False
    # )
    df = pd.read_csv('./data/kdd17/ourpped/AAPL.csv', header=None)

    trading_dates = len(df)
    tra_idx = 0
    val_idx = int(trading_dates*0.7)
    test_idx = int(trading_dates*0.9)
    print(tra_idx, val_idx, test_idx)


    for i in range(val_idx, df.shape[0]):
        X_test.append(df.iloc[i-time_steps:i, :-2])
        # y_train.append((df.iloc[i,-2] + 1) / 2)
        y_test.append(df.iloc[i:i+1, 11])
    X_test, y_test = np.array(X_test), np.array(y_test)

    return X_train, y_train, X_test, y_test


    # data_EOD = []
    # for index, fname in enumerate(fnames):
    #     # print(fname)
    #     single_EOD = np.genfromtxt(
    #         os.path.join(data_path, fname), dtype=float, delimiter=',',
    #         skip_header=False
    #     )
    #     # print('data shape:', single_EOD.shape)
    #     data_EOD.append(single_EOD)
    # fea_dim = data_EOD[0].shape[1] - 2

    # trading_dates = np.genfromtxt(
    #     os.path.join(data_path, '..', 'trading_dates.csv'), dtype=str,
    #     delimiter=',', skip_header=False
    # )\

    # trading_dates = len(data)

    # # print(len(trading_dates), 'trading dates:')

    # # # transform the trading dates into a dictionary with index, at the same
    # # # time, transform the indices into a dictionary with weekdays
    # # dates_index = {}
    # # # indices_weekday = {}
    # # data_wd = np.zeros([len(trading_dates), 5], dtype=float)
    # # wd_encodings = np.identity(5, dtype=float)
    # # for index, date in enumerate(trading_dates):
    # #     dates_index[date] = index
    # #     # indices_weekday[index] = datetime.strptime(date, date_format).weekday()
    # #     data_wd[index] = wd_encodings[datetime.strptime(date, date_format).weekday()]

    # tra_ind = 0
    # val_ind = int(trading_dates*0.7)
    # tes_ind = int(trading_dates*0.9)
    # print(tra_ind, val_ind, tes_ind)

    # tra_num = int(trading_dates*0.7)


    # # generate training, validation, and testing instances
    # # training
    # X_train = np.zeros([-1, time_steps, feature_dim], dtype=float)
    # y_train = np.zeros([])
    # idx = 0
    # for date_ind in range(tra_ind, val_ind):
    #     if date_ind < time_steps:
    #         continue
    #     X_train[idx] = data[date_ind - time_steps: date_ind, : -2]
    #     y_train[idx] = (data[date_ind][-2] + 1) / 2
    #     idx += 1

    # # # validation
    # # val_pv = np.zeros([val_num, seq, fea_dim], dtype=float)
    # # val_wd = np.zeros([val_num, seq, 5], dtype=float)
    # # val_gt = np.zeros([val_num, 1], dtype=float)
    # # ins_ind = 0
    # # for date_ind in range(val_ind, tes_ind):
    # #     # filter out instances without length enough history
    # #     if date_ind < seq:
    # #         continue
    # #     for tic_ind in range(len(fnames)):
    # #         if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8 and \
    # #                         data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
    # #             val_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-2]
    # #             val_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
    # #             val_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
    # #             ins_ind += 1

    # # # testing
    # # tes_pv = np.zeros([tes_num, seq, fea_dim], dtype=float)
    # # tes_wd = np.zeros([tes_num, seq, 5], dtype=float)
    # # tes_gt = np.zeros([tes_num, 1], dtype=float)
    # # ins_ind = 0
    # # for date_ind in range(tes_ind, len(trading_dates)):
    # #     # filter out instances without length enough history
    # #     if date_ind < seq:
    # #         continue
    # #     for tic_ind in range(len(fnames)):
    # #         if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8 and \
    # #                         data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
    # #             tes_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-2]
    # #             # # for the momentum indicator
    # #             # tes_pv[ins_ind, -1, -1] = data_EOD[tic_ind][date_ind - 1, -1] - data_EOD[tic_ind][date_ind - 11, -1]
    # #             tes_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
    # #             tes_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
    # #             ins_ind += 1
    # # return tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt

load_data('./data/index/preprocessed/', 'snp500_p.csv', 11)
# /data/index/preprocessed/snp500_p.csv