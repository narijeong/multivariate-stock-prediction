#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='j01qgzf9'
export NNI_SYS_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/j01qgzf9/trials/CxeQX'
export NNI_TRIAL_JOB_ID='CxeQX'
export NNI_OUTPUT_DIR='/Users/narijeong/Dev/stock-prediction-with-context2/log/j01qgzf9/trials/CxeQX'
export NNI_TRIAL_SEQ_ID='1'
export NNI_CODE_DIR='/Users/narijeong/Dev/stock-prediction-with-context2'
cd $NNI_CODE_DIR
eval 'python pred_nni.py' 1>/Users/narijeong/Dev/stock-prediction-with-context2/log/j01qgzf9/trials/CxeQX/stdout 2>/Users/narijeong/Dev/stock-prediction-with-context2/log/j01qgzf9/trials/CxeQX/stderr
echo $? `date +%s999` >'/Users/narijeong/Dev/stock-prediction-with-context2/log/j01qgzf9/trials/CxeQX/.nni/state'