#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='es8rblhx'
export NNI_SYS_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/es8rblhx/trials/KHCtI'
export NNI_TRIAL_JOB_ID='KHCtI'
export NNI_OUTPUT_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/es8rblhx/trials/KHCtI'
export NNI_TRIAL_SEQ_ID='1'
export NNI_CODE_DIR='/Users/narijeong/Dev/stock-prediction-with-context2'
cd $NNI_CODE_DIR
eval 'python pred_nni.py' 1>/Users/narijeong/Dev/stock-prediction-with-context2/log/es8rblhx/trials/KHCtI/stdout 2>/Users/narijeong/Dev/stock-prediction-with-context2/log/es8rblhx/trials/KHCtI/stderr
echo $? `date +%s999` >'/Users/narijeong/Dev/stock-prediction-with-context2/log/es8rblhx/trials/KHCtI/.nni/state'