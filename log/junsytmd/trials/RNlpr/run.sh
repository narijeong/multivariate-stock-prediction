#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='junsytmd'
export NNI_SYS_DIR='/Users/narijeong/Dev/stock-prediction-with-context/log/junsytmd/trials/RNlpr'
export NNI_TRIAL_JOB_ID='RNlpr'
export NNI_OUTPUT_DIR='/Users/narijeong/Dev/stock-prediction-with-context/log/junsytmd/trials/RNlpr'
export NNI_TRIAL_SEQ_ID='2'
export NNI_CODE_DIR='/Users/narijeong/Dev/stock-prediction-with-context'
cd $NNI_CODE_DIR
eval 'python pred_nni.py' 1>/Users/narijeong/Dev/stock-prediction-with-context/log/junsytmd/trials/RNlpr/stdout 2>/Users/narijeong/Dev/stock-prediction-with-context/log/junsytmd/trials/RNlpr/stderr
echo $? `date +%s999` >'/Users/narijeong/Dev/stock-prediction-with-context/log/junsytmd/trials/RNlpr/.nni/state'