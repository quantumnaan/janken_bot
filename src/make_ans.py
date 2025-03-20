import numpy as np
import pickle as pk
import os

from constatants import *


n = 100 # 人数
m = 20 # 1人あたりのデータ数 
np.random.seed(0) # 乱数固定

ans_transmat = np.random.rand(3, NS, NP)*2-1 # 正解の変換行列, 手の数3, 状態数6, パラメータ次元3
ans_params = np.random.rand(NP, n)*2-1 # 正解のパラメータ

def state_update(choice1, choice2): 
  """
    choice1: 前に出した手
    choice2: 前にCPが(ランダムに)出した手
    return: この時の状態 cf. constatants.py
  """
  win_lose = (choice2 - choice1 + 4) %3 # 0:負け, 1:あいこ, 2:勝ち
  return choice1*3 + win_lose

# データ(手の出し方を表す行列)の生成
sample_data = ans_transmat @ ans_params
sample_data = np.exp(sample_data) / np.sum(np.exp(sample_data), axis=0)

data_ans = {
  "transmat": ans_transmat,
  "params": ans_params,
  "sample_data": sample_data
}

with open(file_data_param, 'wb') as f:
  pk.dump(data_ans, f)
print("data saved")

# データ(実際に出した手)の生成
data_choice = []
for i in range(n):
  data_choice.append([])
  state = np.random.choice(NS)
  for j in range(m):
    choice_ij = np.random.choice(3, p=sample_data[:, state, i])
    data_choice[i].append((state, choice_ij))
    cp_choice = np.random.choice(3)
    state = state_update(choice_ij, cp_choice)

data_choice = np.array(data_choice)
    
with open(file_data, 'wb') as f:
  pk.dump(data_choice, f)