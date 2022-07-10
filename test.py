import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

df = pd.DataFrame({'a':[1,2,3], 'b':[2,3,4]})

test = 1+2