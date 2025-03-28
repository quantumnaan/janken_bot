import numpy as np
import pickle as pk
import os

from constatants import *

def state_update(choice1, choice2): 
  """
    choice1: 前に出した手
    choice2: 前にCPが(ランダムに)出した手
    return: この時の状態 cf. constatants.py
  """
  win_lose = (choice2 - choice1 + 4) %3 # 0:負け, 1:あいこ, 2:勝ち
  return (choice1*3 + win_lose)
  # return choice1

n = 40 # 人数
m = 20 # 1人あたりのデータ数 
np.random.seed(0) # 乱数固定
np.set_printoptions(precision=3)

ans_mu = (np.random.random(NP)-0.5)*5 # 正解の平均
ans_Sig = (np.random.random((NP, NP))-0.5)*3
ans_Sig = ans_Sig @ ans_Sig.T # 正解の共分散行列, 正定値対称行列にする
ans_params = np.random.multivariate_normal(ans_mu, ans_Sig, size=n).T # 正解のパラメータ

print(f"mu: \n{ans_mu}")
print(f"Sig: \n{ans_Sig}")

ans_transmat = (np.random.rand(3, NS, NP)-0.5)*0.4 # 正解の変換行列, 手の数3, 状態数6, パラメータ次元3
ans_transmat[2] = - ans_transmat[0] - ans_transmat[1] # 列和は0(冗長性を削減)

# データ(手の出し方を表す行列)の生成
sample_data = ans_transmat @ ans_params
sample_data = np.exp(sample_data) / np.sum(np.exp(sample_data), axis=0)


data_ans = {
  "transmat": ans_transmat,
  "params": ans_params,
  "sample_data": sample_data,
  "mu": ans_mu,
  "Sig": ans_Sig
}

with open(file_data_param, 'wb') as f:
  pk.dump(data_ans, f)

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
  
print("data saved")