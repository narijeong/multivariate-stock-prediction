#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='wcxp6vn9'
export NNI_SYS_DIR='/Users/narijeong/Dev/stock-prediction-with-attention/log/wcxp6vn9/trials/BcMPa'
export NNI_TRIAL_JOB_ID='BcMPa'
export NNI_OUTPUT_DIR='/Users/narijeong/Dev/stock-prediction-with-attention/log/wcxp6vn9/trials/BcMPa'
export NNI_TRIAL_SEQ_ID='0'
export NNI_CODE_DIR='/Users/narijeong/Dev/stock-prediction-with-attention'
cd $NNI_CODE_DIR
eval 'python pred_nni.py' 1>/Users/narijeong/Dev/stock-prediction-with-attention/log/wcxp6vn9/trials/BcMPa/stdout 2>/Users/narijeong/Dev/stock-prediction-with-attention/log/wcxp6vn9/trials/BcMPa/stderr
echo $? `date +%s999` >'/Users/narijeong/Dev/stock-prediction-with-attention/log/wcxp6vn9/trials/BcMPa/.nni/state'