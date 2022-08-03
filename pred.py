import argparse
import copy
import numpy as np
import os
import random
from sklearn.utils import shuffle
import tensorflow as tf
from time import time
from keras import backend, activations
from keras.layers import Input, Reshape, Dense, LSTM, LayerNormalization, Lambda, Add, Dropout
from keras import Model
from keras.utils.vis_utils import plot_model

from attention import Attention
from multiheadattention import MultiHeadAttention
from load_data import load_cla_data
from evaluator import evaluate

class DTML:
    def __init__(self, data_path, model_path, model_save_path, parameters, steps=1, epochs=50,
                 batch_size=256, gpu=False, tra_date='2014-01-02',
                 val_date='2015-08-03', tes_date='2015-10-01', att=0, hinge=0,
                 fix_init=0, adv=0, reload=0):
        self.data_path = data_path
        self.model_path = model_path
        self.model_save_path = model_save_path
        # model parameters
        self.paras = copy.copy(parameters)
        # training parameters
        self.steps = steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu = gpu

        if att == 1:
            self.att = True
        else:
            self.att = False
        if hinge == 1:
            self.hinge = True
        else:
            self.hinge = False
        if fix_init == 1:
            self.fix_init = True
        else:
            self.fix_init = False
        if adv == 1:
            self.adv_train = True
        else:
            self.adv_train = False
        if reload == 1:
            self.reload = True
        else:
            self.reload = False

        # load data
        self.tra_date = tra_date
        self.val_date = val_date
        self.tes_date = tes_date
        self.tra_pv, self.tra_wd, self.tra_gt, \
        self.val_pv, self.val_wd, self.val_gt, \
        self.tes_pv, self.tes_wd, self.tes_gt = load_cla_data(
            self.data_path,
            tra_date, val_date, tes_date, seq=self.paras['seq']
        )
        self.fea_dim = self.tra_pv.shape[3]

    def get_batch(self, sta_ind=None):
        if sta_ind is None:
            sta_ind = random.randrange(0, self.tra_pv.shape[0])
        if sta_ind + self.batch_size < self.tra_pv.shape[0]:
            end_ind = sta_ind + self.batch_size
        else:
            sta_ind = self.tra_pv.shape[0] - self.batch_size
            end_ind = self.tra_pv.shape[0]
        return self.tra_pv[sta_ind:end_ind, :, :], \
               self.tra_wd[sta_ind:end_ind, :, :], \
               self.tra_gt[sta_ind:end_ind, :]

    def create_model(self):
        print('fea_dim', self.fea_dim)

        def input_reshape(x):
            return backend.reshape(x, (-1, args.seq, self.fea_dim))

        def backend_reshape(x):
            return backend.reshape(x, (-1, self.paras['ticker_num'] + 1, self.fea_dim))

        input_shape = Input(shape=(self.paras['ticker_num']+1, args.seq, self.fea_dim), name='index_input')
        x = Lambda(input_reshape)(input_shape)
        x = Dense(self.fea_dim, activation='tanh')(x)
        
        if args.model == 'lstm':
            x = LSTM(args.unit)(x)

        if args.model == 'dtml':
            print('dtml')
            x = LSTM(args.unit, return_sequences=True)(x)
            x = Attention(self.fea_dim)(x)
            print('attention shape', x.shape)
            x = LayerNormalization()(x)
            x = Lambda(backend_reshape)(x)
            print('backend_reshape shape', x.shape)
            # x = Lambda(lambda x: x + x[:, 0])(x)
            x = Lambda(lambda x: x[:, 1:])(x)
            print('layer shape', x.shape)

            # Data Axis contextx
            mha = MultiHeadAttention(h=8)([x,x,x])
            # Nonlinear Transformation
            add = Add()([x, mha])
            x = Dense(add.shape[-1])(add)
            x = Add()([add, x])
            x = Dropout(rate=0.2)(x)
            x = activations.tanh(x)
            x = Dense(1, activation='sigmoid')(x)

        print('final shape', x.shape)
        model = Model(inputs=input_shape, outputs=x)
        model.compile(loss='mae', optimizer='adam')
        plot_model(model, to_file='./image/model.png', show_shapes=True, show_layer_names=True)
        return model

    def get_latent_rep(self):
        self.construct_graph()

        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        bat_count = self.tra_pv.shape[0] // self.batch_size
        if not (self.tra_pv.shape[0] % self.batch_size == 0):
            bat_count += 1

        tr_lat_rep = np.zeros([bat_count * self.batch_size, self.paras['unit'] * 2],
                              dtype=np.float32)
        tr_gt = np.zeros([bat_count * self.batch_size, 1], dtype=np.float32)
        for j in range(bat_count):
            pv_b, wd_b, gt_b = self.get_batch(j * self.batch_size)
            feed_dict = {
                self.pv_var: pv_b,
                self.wd_var: wd_b,
                self.gt_var: gt_b
            }
            lat_rep, cur_obj, cur_loss, cur_l2, cur_al = sess.run(
                (self.fea_con, self.obj_func, self.loss, self.l2_norm,
                 self.adv_loss),
                feed_dict
            )
            print(lat_rep.shape)
            tr_lat_rep[j * self.batch_size: (j + 1) * self.batch_size, :] = lat_rep
            tr_gt[j * self.batch_size: (j + 1) * self.batch_size,:] = gt_b

        # test on validation set
        feed_dict = {
            self.pv_var: self.val_pv,
            self.wd_var: self.val_wd,
            self.gt_var: self.val_gt
        }
        val_loss, val_lat_rep, val_pre = sess.run(
            (self.loss, self.fea_con, self.pred), feed_dict
        )
        cur_val_perf = evaluate(val_pre, self.val_gt, self.hinge)
        print('\tVal per:', cur_val_perf)

        sess.close()
        tf.reset_default_graph()
        np.savetxt(self.model_save_path + '_val_lat_rep.csv', val_lat_rep)
        np.savetxt(self.model_save_path + '_tr_lat_rep.csv', tr_lat_rep)
        np.savetxt(self.model_save_path + '_val_gt.csv', self.val_gt)
        np.savetxt(self.model_save_path + '_tr_gt.csv', tr_gt)

    def predict_adv(self):

        self.construct_graph()

        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        bat_count = self.tra_pv.shape[0] // self.batch_size
        if not (self.tra_pv.shape[0] % self.batch_size == 0):
            bat_count += 1
        tra_perf = None
        adv_perf = None
        for j in range(bat_count):
            pv_b, wd_b, gt_b = self.get_batch(j * self.batch_size)
            feed_dict = {
                self.pv_var: pv_b,
                self.wd_var: wd_b,
                self.gt_var: gt_b
            }
            cur_pre, cur_adv_pre, cur_obj, cur_loss, cur_l2, cur_al = sess.run(
                (self.pred, self.adv_pred, self.obj_func, self.loss, self.l2_norm,
                 self.adv_loss),
                feed_dict
            )
            cur_tra_perf = evaluate(cur_pre, gt_b, self.hinge)
            cur_adv_perf = evaluate(cur_adv_pre, gt_b, self.hinge)
            if tra_perf is None:
                tra_perf = copy.copy(cur_tra_perf)
            else:
                for metric in tra_perf.keys():
                    tra_perf[metric] = tra_perf[metric] + cur_tra_perf[metric]
            if adv_perf is None:
                adv_perf = copy.copy(cur_adv_perf)
            else:
                for metric in adv_perf.keys():
                    adv_perf[metric] = adv_perf[metric] + cur_adv_perf[metric]
        for metric in tra_perf.keys():
            tra_perf[metric] = tra_perf[metric] / bat_count
            adv_perf[metric] = adv_perf[metric] / bat_count

        print('Clean samples performance:', tra_perf)
        print('Adversarial samples performance:', adv_perf)

        # test on validation set
        feed_dict = {
            self.pv_var: self.val_pv,
            self.wd_var: self.val_wd,
            self.gt_var: self.val_gt
        }
        val_loss, val_pre, val_adv_pre = sess.run(
            (self.loss, self.pred, self.adv_pred), feed_dict
        )
        cur_valid_perf = evaluate(val_pre, self.val_gt, self.hinge)
        print('\tVal per clean:', cur_valid_perf)
        adv_valid_perf = evaluate(val_adv_pre, self.val_gt, self.hinge)
        print('\tVal per adversarial:', adv_valid_perf)

        # test on testing set
        feed_dict = {
            self.pv_var: self.tes_pv,
            self.wd_var: self.tes_wd,
            self.gt_var: self.tes_gt
        }
        test_loss, tes_pre, tes_adv_pre = sess.run(
            (self.loss, self.pred, self.adv_pred), feed_dict
        )
        cur_test_perf = evaluate(tes_pre, self.tes_gt, self.hinge)
        print('\tTest per clean:', cur_test_perf)
        adv_test_perf = evaluate(tes_adv_pre, self.tes_gt, self.hinge)
        print('\tTest per adversarial:', adv_test_perf)

        sess.close()
        tf.reset_default_graph()

    def predict_record(self):
        self.construct_graph()

        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        # test on validation set
        feed_dict = {
            self.pv_var: self.val_pv,
            self.wd_var: self.val_wd,
            self.gt_var: self.val_gt
        }
        val_loss, val_pre = sess.run(
            (self.loss, self.pred), feed_dict
        )
        cur_valid_perf = evaluate(val_pre, self.val_gt, self.hinge)
        print('\tVal per:', cur_valid_perf, '\tVal loss:', val_loss)
        np.savetxt(self.model_save_path + '_val_prediction.csv', val_pre)

        # test on testing set
        feed_dict = {
            self.pv_var: self.tes_pv,
            self.wd_var: self.tes_wd,
            self.gt_var: self.tes_gt
        }
        test_loss, tes_pre = sess.run(
            (self.loss, self.pred), feed_dict
        )
        cur_test_perf = evaluate(tes_pre, self.tes_gt, self.hinge)
        print('\tTest per:', cur_test_perf, '\tTest loss:', test_loss)
        np.savetxt(self.model_save_path + '_tes_prediction.csv', tes_pre)
        sess.close()
        tf.reset_default_graph()

    def test(self):
        self.construct_graph()

        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        # test on validation set
        feed_dict = {
            self.pv_var: self.val_pv,
            self.wd_var: self.val_wd,
            self.gt_var: self.val_gt
        }
        val_loss, val_pre = sess.run(
            (self.loss, self.pred), feed_dict
        )
        cur_valid_perf = evaluate(val_pre, self.val_gt, self.hinge)
        print('\tVal per:', cur_valid_perf, '\tVal loss:', val_loss)

        # test on testing set
        feed_dict = {
            self.pv_var: self.tes_pv,
            self.wd_var: self.tes_wd,
            self.gt_var: self.tes_gt
        }
        test_loss, tes_pre = sess.run(
            (self.loss, self.pred), feed_dict
        )
        cur_test_perf = evaluate(tes_pre, self.tes_gt, self.hinge)
        print('\tTest per:', cur_test_perf, '\tTest loss:', test_loss)
        sess.close()
        tf.reset_default_graph()

    def train(self, tune_para=False):

        model = self.create_model()
        print('tra_pv', self.tra_pv.shape)
        print('tra_gt', self.tra_gt.shape)
        if args.model == 'lstm':
            model.fit(x=self.tra_pv, y=self.tra_gt ,epochs=1)
        
        if args.model == 'dtml':
            model.fit(x =self.tra_pv, y=self.tra_gt ,epochs=1)

        # tes_pred = model.predict(self.tes_pv)
        # test_perf = evaluate(tes_pred, self.tes_gt)
        # print('\tTest per:', test_perf)


        # best_valid_pred = np.zeros(self.val_gt.shape, dtype=float)
        # best_test_pred = np.zeros(self.tes_gt.shape, dtype=float)

        # best_valid_perf = {
        #     'acc': 0, 'mcc': -2
        # }
        # best_test_perf = {
        #     'acc': 0, 'mcc': -2
        # }

        # bat_count = self.tra_pv.shape[0] // self.batch_size
        # if not (self.tra_pv.shape[0] % self.batch_size == 0):
        #     bat_count += 1
        # for i in range(self.epochs):
        #     t1 = time()
        #     # first_batch = True
        #     tra_loss = 0.0
        #     tra_obj = 0.0
        #     l2 = 0.0
        #     tra_adv = 0.0
        #     for j in range(bat_count):
        #         pv_b, wd_b, gt_b = self.get_batch(j * self.batch_size)
        #         feed_dict = {
        #             self.pv_var: pv_b,
        #             self.wd_var: wd_b,
        #             self.gt_var: gt_b
        #         }
        #         cur_pre, cur_obj, cur_loss, cur_l2, cur_al, batch_out = sess.run(
        #             (self.pred, self.obj_func, self.loss, self.l2_norm, self.adv_loss,
        #              self.optimizer),
        #             feed_dict
        #         )

        #         tra_loss += cur_loss
        #         tra_obj += cur_obj
        #         l2 += cur_l2
        #         tra_adv += cur_al
        #     print('----->>>>> Training:', tra_obj / bat_count,
        #           tra_loss / bat_count, l2 / bat_count, tra_adv / bat_count)

        #     if not tune_para:
        #         tra_loss = 0.0
        #         tra_obj = 0.0
        #         l2 = 0.0
        #         tra_acc = 0.0
        #         for j in range(bat_count):
        #             pv_b, wd_b, gt_b = self.get_batch(
        #                 j * self.batch_size)
        #             feed_dict = {
        #                 self.pv_var: pv_b,
        #                 self.wd_var: wd_b,
        #                 self.gt_var: gt_b
        #             }
        #             cur_obj, cur_loss, cur_l2, cur_pre = sess.run(
        #                 (self.obj_func, self.loss, self.l2_norm, self.pred),
        #                 feed_dict
        #             )
        #             cur_tra_perf = evaluate(cur_pre, gt_b, self.hinge)
        #             tra_loss += cur_loss
        #             l2 += cur_l2
        #             tra_obj += cur_obj
        #             tra_acc += cur_tra_perf['acc']
        #         print('Training:', tra_obj / bat_count, tra_loss / bat_count,
        #               l2 / bat_count, '\tTrain per:', tra_acc / bat_count)

        #     # test on validation set
        #     feed_dict = {
        #         self.pv_var: self.val_pv,
        #         self.wd_var: self.val_wd,
        #         self.gt_var: self.val_gt
        #     }
        #     val_loss, val_pre = sess.run(
        #         (self.loss, self.pred), feed_dict
        #     )
        #     cur_valid_perf = evaluate(val_pre, self.val_gt, self.hinge)
        #     print('\tVal per:', cur_valid_perf, '\tVal loss:', val_loss)

        #     # test on testing set
        #     feed_dict = {
        #         self.pv_var: self.tes_pv,
        #         self.wd_var: self.tes_wd,
        #         self.gt_var: self.tes_gt
        #     }
        #     test_loss, tes_pre = sess.run(
        #         (self.loss, self.pred), feed_dict
        #     )
        #     cur_test_perf = evaluate(tes_pre, self.tes_gt, self.hinge)
        #     print('\tTest per:', cur_test_perf, '\tTest loss:', test_loss)

        #     if cur_valid_perf['acc'] > best_valid_perf['acc']:
        #         best_valid_perf = copy.copy(cur_valid_perf)
        #         best_valid_pred = copy.copy(val_pre)
        #         best_test_perf = copy.copy(cur_test_perf)
        #         best_test_pred = copy.copy(tes_pre)
        #         if not tune_para:
        #             saver.save(sess, self.model_save_path)
        #     self.tra_pv, self.tra_wd, self.tra_gt = shuffle(
        #         self.tra_pv, self.tra_wd, self.tra_gt, random_state=0
        #     )
        #     t4 = time()
        #     print('epoch:', i, ('time: %.4f ' % (t4 - t1)))
        # print('\nBest Valid performance:', best_valid_perf)
        # print('\tBest Test performance:', best_test_perf)
        # sess.close()
        # tf.reset_default_graph()
        # if tune_para:
        #     return best_valid_perf, best_test_perf
        # return best_valid_pred, best_test_pred

    def update_model(self, parameters):
        data_update = False
        if not parameters['seq'] == self.paras['seq']:
            data_update = True
        for name, value in parameters.items():
            self.paras[name] = value
        if data_update:
            self.tra_pv, self.tra_wd, self.tra_gt, \
            self.val_pv, self.val_wd, self.val_gt, \
            self.tes_pv, self.tes_wd, self.tes_gt = load_cla_data(
                self.data_path,
                self.tra_date, self.val_date, self.tes_date, seq=self.paras['seq']
            )
        return True

if __name__ == '__main__':
    desc = 'the lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', '--path', help='path of pv data', type=str,
                        default='./data/stocknet-dataset/price/ourpped')
    parser.add_argument('-l', '--seq', help='length of history', type=int,
                        default=5)
    parser.add_argument('-u', '--unit', help='number of hidden units in lstm',
                        type=int, default=32)
    parser.add_argument('-l2', '--alpha_l2', type=float, default=1e-2,
                        help='alpha for l2 regularizer')
    parser.add_argument('-la', '--beta_adv', type=float, default=1e-2,
                        help='beta for adverarial loss')
    parser.add_argument('-le', '--epsilon_adv', type=float, default=1e-2,
                        help='epsilon to control the scale of noise')
    parser.add_argument('-s', '--step', help='steps to make prediction',
                        type=int, default=1)
    parser.add_argument('-b', '--batch_size', help='batch size', type=int,
                        default=1024)
    parser.add_argument('-e', '--epoch', help='epoch', type=int, default=150)
    parser.add_argument('-r', '--learning_rate', help='learning rate',
                        type=float, default=1e-2)
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    parser.add_argument('-q', '--model_path', help='path to load model',
                        type=str, default='./saved_model/acl18_alstm/exp')
    parser.add_argument('-qs', '--model_save_path', type=str, help='path to save model',
                        default='./tmp/model')
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, pred')
    parser.add_argument('-m', '--model', type=str, default='lstm',
                        help='lstm, att_lstm, dtml, di_lstm, att_lstm, week_lstm, aw_lstm')
    parser.add_argument('-f', '--fix_init', type=int, default=0,
                        help='use fixed initialization')
    parser.add_argument('-a', '--att', type=int, default=1,
                        help='use attention model')
    parser.add_argument('-w', '--week', type=int, default=0,
                        help='use week day data')
    parser.add_argument('-v', '--adv', type=int, default=0,
                        help='adversarial training')
    parser.add_argument('-hi', '--hinge_lose', type=int, default=1,
                        help='use hinge lose')
    parser.add_argument('-rl', '--reload', type=int, default=0,
                        help='use pre-trained parameters')
    args = parser.parse_args()
    print(args)

    parameters = {
        'seq': int(args.seq),
        'unit': int(args.unit),
        'alp': float(args.alpha_l2),
        'bet': float(args.beta_adv),
        'eps': float(args.epsilon_adv),
        'lr': float(args.learning_rate)
    }

    fnames = [fname for fname in os.listdir(args.path) if
              os.path.isfile(os.path.join(args.path, fname))]
    parameters['ticker_num'] = len(fnames) - 1

    if 'stocknet' in args.path:
        tra_date = '2014-01-02'
        val_date = '2015-08-03'
        tes_date = '2015-10-01'
    elif 'kdd17' in args.path:
        tra_date = '2007-01-03'
        val_date = '2015-01-02'
        tes_date = '2016-01-04'
    else:
        print('unexpected path: %s' % args.path)
        exit(0)


    

    model = DTML(
        data_path=args.path,
        model_path=args.model_path,
        model_save_path=args.model_save_path,
        parameters=parameters,
        steps=args.step,
        epochs=args.epoch, batch_size=args.batch_size, gpu=args.gpu,
        tra_date=tra_date, val_date=val_date, tes_date=tes_date, att=args.att,
        hinge=args.hinge_lose, fix_init=args.fix_init, adv=args.adv,
        reload=args.reload
    )

    if args.action == 'train':
        model.train()
    # elif args.action == 'test':
    #     pure_LSTM.test()
    # elif args.action == 'report':
    #     for i in range(5):
    #         pure_LSTM.train()
    # elif args.action == 'pred':
    #     pure_LSTM.predict_record()
    # elif args.action == 'adv':
    #     pure_LSTM.predict_adv()
    # elif args.action == 'latent':
    #     pure_LSTM.get_latent_rep()