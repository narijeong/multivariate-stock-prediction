#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='cfr649mb'
export NNI_SYS_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/cfr649mb/trials/bRol9'
export NNI_TRIAL_JOB_ID='bRol9'
export NNI_OUTPUT_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/cfr649mb/trials/bRol9'
export NNI_TRIAL_SEQ_ID='2'
export NNI_CODE_DIR='/Users/narijeong/Dev/stock-prediction-with-context2'
cd $NNI_CODE_DIR
eval 'python pred_nni.py' 1>/Users/narijeong/Dev/stock-prediction-with-context2/log/cfr649mb/trials/bRol9/stdout 2>/Users/narijeong/Dev/stock-prediction-with-context2/log/cfr649mb/trials/bRol9/stderr
echo $? `date +%s999` >'/Users/narijeong/Dev/stock-prediction-with-context2/log/cfr649mb/trials/bRol9/.nni/state'