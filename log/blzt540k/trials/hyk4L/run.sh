#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='blzt540k'
export NNI_SYS_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/blzt540k/trials/hyk4L'
export NNI_TRIAL_JOB_ID='hyk4L'
export NNI_OUTPUT_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/blzt540k/trials/hyk4L'
export NNI_TRIAL_SEQ_ID='0'
export NNI_CODE_DIR='/Users/narijeong/Dev/stock-prediction-with-context2'
cd $NNI_CODE_DIR
eval 'python pred_nni.py' 1>/Users/narijeong/Dev/stock-prediction-with-context2/log/blzt540k/trials/hyk4L/stdout 2>/Users/narijeong/Dev/stock-prediction-with-context2/log/blzt540k/trials/hyk4L/stderr
echo $? `date +%s999` >'/Users/narijeong/Dev/stock-prediction-with-context2/log/blzt540k/trials/hyk4L/.nni/state'