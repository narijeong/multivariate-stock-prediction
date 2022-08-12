# Accurate Multivariate Stock Movement Prediction via Data-Axis Transformer with Multi-Level Contexts Implementation
This is a Keras implementation of the stock movement prediction model in "Accurate Multivariate Stock Movement Prediction via Data-Axis Transformer with Multi-Level Contexts" (Jaemin Yoo, Yejun Soun, Yong-chan Park, and U Kang, KDD, 2021).  

![DTML architecture](./imgage/architecture.png)
### The model is consisted of 3 stages
- time axis attention
- context aggregation
- data axis attention

### The model presented compares the performance with two other models 
- https://www.ijcai.org/proceedings/2019/0810.pdf
- Stocknet: https://github.com/yumoxu/stocknet-code
### how to run
see hyperparameters
### installation
brew install graphviz
pip install -r requirements.txt
### add kernel
'''
ipython kernel install --user --name=ml
'''

### start NNI UI
```
nnictl create --config config.yaml
```

### stop NNI 
```
nnictl stop --all
```

### NNI 관련 파일
1. config.yaml (nni 설정파일)
2. pred_nni.py (모델 생성및 nni hyper parameter 지정)


### NNI run
- threshold 사용하여 (하락, threshold 구간 = 0) , (상승 = 1) 결과 = VmMI9