#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='fxrcn0h5'
export NNI_SYS_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/fxrcn0h5/trials/OGvrG'
export NNI_TRIAL_JOB_ID='OGvrG'
export NNI_OUTPUT_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/fxrcn0h5/trials/OGvrG'
export NNI_TRIAL_SEQ_ID='1'
export NNI_CODE_DIR='/Users/narijeong/Dev/stock-prediction-with-context2'
cd $NNI_CODE_DIR
eval 'python pred_nni.py' 1>/Users/narijeong/Dev/stock-prediction-with-context2/log/fxrcn0h5/trials/OGvrG/stdout 2>/Users/narijeong/Dev/stock-prediction-with-context2/log/fxrcn0h5/trials/OGvrG/stderr
echo $? `date +%s999` >'/Users/narijeong/Dev/stock-prediction-with-context2/log/fxrcn0h5/trials/OGvrG/.nni/state'