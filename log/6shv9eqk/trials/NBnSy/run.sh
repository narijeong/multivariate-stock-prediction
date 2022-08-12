#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='6shv9eqk'
export NNI_SYS_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/6shv9eqk/trials/NBnSy'
export NNI_TRIAL_JOB_ID='NBnSy'
export NNI_OUTPUT_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/6shv9eqk/trials/NBnSy'
export NNI_TRIAL_SEQ_ID='2'
export NNI_CODE_DIR='/Users/narijeong/Dev/stock-prediction-with-context2'
cd $NNI_CODE_DIR
eval 'python pred_nni.py' 1>/Users/narijeong/Dev/stock-prediction-with-context2/log/6shv9eqk/trials/NBnSy/stdout 2>/Users/narijeong/Dev/stock-prediction-with-context2/log/6shv9eqk/trials/NBnSy/stderr
echo $? `date +%s999` >'/Users/narijeong/Dev/stock-prediction-with-context2/log/6shv9eqk/trials/NBnSy/.nni/state'