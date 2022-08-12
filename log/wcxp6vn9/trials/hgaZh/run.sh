#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='wcxp6vn9'
export NNI_SYS_DIR='/Users/narijeong/Dev/stock-prediction-with-attention/log/wcxp6vn9/trials/hgaZh'
export NNI_TRIAL_JOB_ID='hgaZh'
export NNI_OUTPUT_DIR='/Users/narijeong/Dev/stock-prediction-with-attention/log/wcxp6vn9/trials/hgaZh'
export NNI_TRIAL_SEQ_ID='2'
export NNI_CODE_DIR='/Users/narijeong/Dev/stock-prediction-with-attention'
cd $NNI_CODE_DIR
eval 'python pred_nni.py' 1>/Users/narijeong/Dev/stock-prediction-with-attention/log/wcxp6vn9/trials/hgaZh/stdout 2>/Users/narijeong/Dev/stock-prediction-with-attention/log/wcxp6vn9/trials/hgaZh/stderr
echo $? `date +%s999` >'/Users/narijeong/Dev/stock-prediction-with-attention/log/wcxp6vn9/trials/hgaZh/.nni/state'