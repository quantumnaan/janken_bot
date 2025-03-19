import numpy as np
import pickle as pk
import os

file_data = "./data/sample_data.pkl"

n = 100 # 人数
np.random.seed(0) # 乱数固定

ans_transmat = np.random.rand(3, 6, 3)*4-2 # 正解の変換行列, 手の数3, 状態数6, パラメータ次元3
ans_params = np.random.rand(3, n)*4-2 # 正解のパラメータ
# データの生成
sample_data = ans_transmat @ ans_params
sample_data = np.exp(sample_data) / np.sum(np.exp(sample_data), axis=0)

data = {
  "transmat": ans_transmat,
  "params": ans_params,
  "sample_data": sample_data
}

with open(file_data, 'wb') as f:
  pk.dump(data, f)
print("data saved")
