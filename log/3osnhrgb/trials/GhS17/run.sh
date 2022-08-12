#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='3osnhrgb'
export NNI_SYS_DIR='/Users/narijeong/Dev/stock-prediction-with-attention/log/3osnhrgb/trials/GhS17'
export NNI_TRIAL_JOB_ID='GhS17'
export NNI_OUTPUT_DIR='/Users/narijeong/Dev/stock-prediction-with-attention/log/3osnhrgb/trials/GhS17'
export NNI_TRIAL_SEQ_ID='1'
export NNI_CODE_DIR='/Users/narijeong/Dev/stock-prediction-with-attention'
cd $NNI_CODE_DIR
eval 'python pred_nni.py' 1>/Users/narijeong/Dev/stock-prediction-with-attention/log/3osnhrgb/trials/GhS17/stdout 2>/Users/narijeong/Dev/stock-prediction-with-attention/log/3osnhrgb/trials/GhS17/stderr
echo $? `date +%s999` >'/Users/narijeong/Dev/stock-prediction-with-attention/log/3osnhrgb/trials/GhS17/.nni/state'