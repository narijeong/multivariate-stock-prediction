#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='p3vsq0uy'
export NNI_SYS_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/p3vsq0uy/trials/LAwfa'
export NNI_TRIAL_JOB_ID='LAwfa'
export NNI_OUTPUT_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/p3vsq0uy/trials/LAwfa'
export NNI_TRIAL_SEQ_ID='0'
export NNI_CODE_DIR='/Users/narijeong/Dev/stock-prediction-with-context2'
cd $NNI_CODE_DIR
eval 'python pred_nni.py' 1>/Users/narijeong/Dev/stock-prediction-with-context2/log/p3vsq0uy/trials/LAwfa/stdout 2>/Users/narijeong/Dev/stock-prediction-with-context2/log/p3vsq0uy/trials/LAwfa/stderr
echo $? `date +%s999` >'/Users/narijeong/Dev/stock-prediction-with-context2/log/p3vsq0uy/trials/LAwfa/.nni/state'