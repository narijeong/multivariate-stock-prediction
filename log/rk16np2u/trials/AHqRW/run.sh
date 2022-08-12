#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='rk16np2u'
export NNI_SYS_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/rk16np2u/trials/AHqRW'
export NNI_TRIAL_JOB_ID='AHqRW'
export NNI_OUTPUT_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/rk16np2u/trials/AHqRW'
export NNI_TRIAL_SEQ_ID='2'
export NNI_CODE_DIR='/Users/narijeong/Dev/stock-prediction-with-context2'
cd $NNI_CODE_DIR
eval 'python pred_nni.py' 1>/Users/narijeong/Dev/stock-prediction-with-context2/log/rk16np2u/trials/AHqRW/stdout 2>/Users/narijeong/Dev/stock-prediction-with-context2/log/rk16np2u/trials/AHqRW/stderr
echo $? `date +%s999` >'/Users/narijeong/Dev/stock-prediction-with-context2/log/rk16np2u/trials/AHqRW/.nni/state'