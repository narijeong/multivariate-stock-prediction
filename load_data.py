from datetime import datetime
import numpy as np
import os

# return data in form (batch size, num stocks, time_steps, feature_size)
def load_cla_data(data_path, tra_date, val_date, tes_date, seq=2,
                  date_format='%Y-%m-%d'):
    fnames = [fname for fname in os.listdir(data_path) if
              os.path.isfile(os.path.join(data_path, fname))]
    print(len(fnames), ' tickers selected')
    ticker_num = len(fnames)
    
    fnames.remove('SNP500.csv')
    fnames.insert(0, 'SNP500.csv')

    data_EOD = []
    for index, fname in enumerate(fnames):
        # print(fname)
        single_EOD = np.genfromtxt(
            os.path.join(data_path, fname), dtype=float, delimiter=',',
            skip_header=False
        )
        # print('data shape:', single_EOD.shape)
        data_EOD.append(single_EOD)
    fea_dim = data_EOD[0].shape[1] - 2

    trading_dates = np.genfromtxt(
        os.path.join(data_path, '..', 'trading_dates.csv'), dtype=str,
        delimiter=',', skip_header=False
    )
    print(len(trading_dates), 'trading dates')

    # transform the trading dates into a dictionary with index, at the same
    # time, transform the indices into a dictionary with weekdays
    dates_index = {}
    # indices_weekday = {}
    data_wd = np.zeros([len(trading_dates), 5], dtype=float)
    wd_encodings = np.identity(5, dtype=float)
    for index, date in enumerate(trading_dates):
        dates_index[date] = index
        # indices_weekday[index] = datetime.strptime(date, date_format).weekday()
        data_wd[index] = wd_encodings[datetime.strptime(date, date_format).weekday()]

    tra_ind = dates_index[tra_date]
    val_ind = dates_index[val_date]
    tes_ind = dates_index[tes_date]
    print('tra, val, tes index: ', tra_ind, val_ind, tes_ind)

    # count training, validation, and testing instances
    tra_num = 0
    val_num = 0
    tes_num = 0
    # training
    for date_ind in range(tra_ind, val_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                tra_num += 1
    print(tra_num, ' training instances')

    # validation
    for date_ind in range(val_ind, tes_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                val_num += 1
    print(val_num, ' validation instances')

    # testing
    for date_ind in range(tes_ind, len(trading_dates)):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                tes_num += 1
    print(tes_num, ' testing instances')

    # generate training, validation, and testing instances
    # training
    tra_pv = np.zeros([tra_num, ticker_num, seq, fea_dim], dtype=float)
    tra_wd = np.zeros([tra_num, seq, 5], dtype=float)
    tra_gt = np.zeros([tra_num, ticker_num-1], dtype=float)
    ins_ind = 0
    for date_ind in range(tra_ind, val_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                tra_pv[ins_ind][tic_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, : -2]
                tra_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                if tic_ind != 0:
                    tra_gt[ins_ind, tic_ind-1] = data_EOD[tic_ind][date_ind][-2]
                ins_ind += 1

    # validation
    val_pv = np.zeros([val_num, ticker_num, seq, fea_dim], dtype=float)
    val_wd = np.zeros([val_num, seq, 5], dtype=float)
    val_gt = np.zeros([val_num, ticker_num-1], dtype=float)
    ins_ind = 0
    for date_ind in range(val_ind, tes_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                val_pv[ins_ind][tic_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-2]
                val_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                if tic_ind != 0:
                    val_gt[ins_ind, tic_ind-1] = data_EOD[tic_ind][date_ind][-2]
                ins_ind += 1

    # testing
    tes_pv = np.zeros([tes_num, ticker_num, seq, fea_dim], dtype=float)
    tes_wd = np.zeros([tes_num, seq, 5], dtype=float)
    tes_gt = np.zeros([tes_num, ticker_num-1], dtype=float)
    ins_ind = 0
    for date_ind in range(tes_ind, len(trading_dates)):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                tes_pv[ins_ind][tic_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-2]
                # # for the momentum indicator
                # tes_pv[ins_ind, -1, -1] = data_EOD[tic_ind][date_ind - 1, -1] - data_EOD[tic_ind][date_ind - 11, -1]
                tes_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                if tic_ind != 0:
                    tes_gt[ins_ind, tic_ind-1] = data_EOD[tic_ind][date_ind][-2]
                ins_ind += 1        

    return tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt

if __name__ == '__main__':
    # TEST
    _, _, tra_gt, _, _, val_gt, _, _, tes_gt = load_cla_data(
        './data/kdd17/ourpped/',
        '2007-01-03', '2015-01-02', '2016-01-04'
    )
    print(np.sum(tra_gt))
    print(np.sum(val_gt))
    print(np.sum(tes_gt))
    print(np.sum(tes_gt) / 3720)

# return data in form (batch size, time_steps, feature_size)
def load_cla_data2(data_path, tra_date, val_date, tes_date, seq=2,
                  date_format='%Y-%m-%d'):
    fnames = [fname for fname in os.listdir(data_path) if
              os.path.isfile(os.path.join(data_path, fname))]
    print(len(fnames), ' tickers selected')

    fnames.remove('SNP500.csv')
    fnames.insert(0, 'SNP500.csv')

    data_EOD = []
    for index, fname in enumerate(fnames):
        # print(fname)
        single_EOD = np.genfromtxt(
            os.path.join(data_path, fname), dtype=float, delimiter=',',
            skip_header=False
        )
        # print('data shape:', single_EOD.shape)
        data_EOD.append(single_EOD)
    fea_dim = data_EOD[0].shape[1] - 2

    trading_dates = np.genfromtxt(
        os.path.join(data_path, '..', 'trading_dates.csv'), dtype=str,
        delimiter=',', skip_header=False
    )
    print(len(trading_dates), 'trading dates:')

    # transform the trading dates into a dictionary with index, at the same
    # time, transform the indices into a dictionary with weekdays
    dates_index = {}
    # indices_weekday = {}
    data_wd = np.zeros([len(trading_dates), 5], dtype=float)
    wd_encodings = np.identity(5, dtype=float)
    for index, date in enumerate(trading_dates):
        dates_index[date] = index
        # indices_weekday[index] = datetime.strptime(date, date_format).weekday()
        data_wd[index] = wd_encodings[datetime.strptime(date, date_format).weekday()]

    tra_ind = dates_index[tra_date]
    val_ind = dates_index[val_date]
    tes_ind = dates_index[tes_date]
    print(tra_ind, val_ind, tes_ind)

    # count training, validation, and testing instances
    def count_instances(start_ind, end_ind):
        num = 0
        for date_ind in range(start_ind, end_ind):
            # filter out instances without length enough history
            if date_ind < seq:
                continue
            for tic_ind in range(len(fnames)):
                if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                    num += 1
        return num
    tra_num = count_instances(tra_ind, val_ind)
    val_num = count_instances(val_ind, tes_ind)
    tes_num = count_instances(tes_ind, len(trading_dates))
    print(tra_num, ' training instances')
    print(val_num, ' validaing instances')
    print(tes_num, ' testing instances')

    # generate training, validation, and testing instances
    # training
    tra_pv = np.zeros([tra_num, seq, fea_dim], dtype=float)
    tra_wd = np.zeros([tra_num, seq, 5], dtype=float)
    tra_gt = np.zeros([tra_num, 1], dtype=float)
    ins_ind = 0
    for date_ind in range(tra_ind, val_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                tra_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, : -2]
                tra_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                tra_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                ins_ind += 1
    
    # validation
    val_pv = np.zeros([val_num, seq, fea_dim], dtype=float)
    val_wd = np.zeros([val_num, seq, 5], dtype=float)
    val_gt = np.zeros([val_num, 1], dtype=float)
    ins_ind = 0
    for date_ind in range(val_ind, tes_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                val_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-2]
                val_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                val_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                ins_ind += 1

    # testing
    tes_pv = np.zeros([tes_num, seq, fea_dim], dtype=float)
    tes_wd = np.zeros([tes_num, seq, 5], dtype=float)
    tes_gt = np.zeros([tes_num, 1], dtype=float)
    ins_ind = 0
    for date_ind in range(tes_ind, len(trading_dates)):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                tes_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-2]
                # # for the momentum indicator
                # tes_pv[ins_ind, -1, -1] = data_EOD[tic_ind][date_ind - 1, -1] - data_EOD[tic_ind][date_ind - 11, -1]
                tes_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                tes_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                ins_ind += 1

    return tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt

if __name__ == '__main__':
    # TEST
    _, _, tra_gt, _, _, val_gt, _, _, tes_gt = load_cla_data(
        './data/kdd17/ourpped/',
        '2007-01-03', '2015-01-02', '2016-01-04'
    )
    print(np.sum(tra_gt))
    print(np.sum(val_gt))
    print(np.sum(tes_gt))
    print(np.sum(tes_gt) / 3720)