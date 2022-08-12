#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='rc7kwaq0'
export NNI_SYS_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/rc7kwaq0/trials/uW1oQ'
export NNI_TRIAL_JOB_ID='uW1oQ'
export NNI_OUTPUT_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/rc7kwaq0/trials/uW1oQ'
export NNI_TRIAL_SEQ_ID='2'
export NNI_CODE_DIR='/Users/narijeong/Dev/stock-prediction-with-context2'
cd $NNI_CODE_DIR
eval 'python pred_nni.py' 1>/Users/narijeong/Dev/stock-prediction-with-context2/log/rc7kwaq0/trials/uW1oQ/stdout 2>/Users/narijeong/Dev/stock-prediction-with-context2/log/rc7kwaq0/trials/uW1oQ/stderr
echo $? `date +%s999` >'/Users/narijeong/Dev/stock-prediction-with-context2/log/rc7kwaq0/trials/uW1oQ/.nni/state'