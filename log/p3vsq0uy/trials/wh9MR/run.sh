#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='p3vsq0uy'
export NNI_SYS_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/p3vsq0uy/trials/wh9MR'
export NNI_TRIAL_JOB_ID='wh9MR'
export NNI_OUTPUT_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/p3vsq0uy/trials/wh9MR'
export NNI_TRIAL_SEQ_ID='2'
export NNI_CODE_DIR='/Users/narijeong/Dev/stock-prediction-with-context2'
cd $NNI_CODE_DIR
eval 'python pred_nni.py' 1>/Users/narijeong/Dev/stock-prediction-with-context2/log/p3vsq0uy/trials/wh9MR/stdout 2>/Users/narijeong/Dev/stock-prediction-with-context2/log/p3vsq0uy/trials/wh9MR/stderr
echo $? `date +%s999` >'/Users/narijeong/Dev/stock-prediction-with-context2/log/p3vsq0uy/trials/wh9MR/.nni/state'