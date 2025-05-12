import numpy as np
import pickle as pk
import os
import csv

from constatants import *
from utils import *

"""
file_data_param には以下のデータを保存する
  - 正解の変換行列
  - 正解のパラメータ
  - 正解の平均
  - 正解の共分散行列

file_data には以下のデータを保存する
  - [人i:[ターンt:(人iが出した手, cpが出した手)]]
"""

  # return choice1

n = 100 # 人数
m = 20 # 1人あたりのデータ数 
np.random.seed(0) # 乱数固定
np.set_printoptions(precision=3)


zs = np.random.randn(NP, n)

transmat = (np.random.rand(3, NS, NP)-0.5) # 正解の変換行列, 手の数3, 状態数6, パラメータ次元ZDIM
transmat = transmat * 2

true_mats = transmat @ zs
true_mats = np.exp(true_mats)
true_mats = true_mats / np.sum(true_mats, axis=0)
data_players = []

for i in range(n):
    state = np.random.randint(0, 9)
    data = []
    for j in range(m):
        choice_cpu = np.random.randint(0, 3)
        choice_player = np.random.choice([0,1,2], p=true_mats[:,state,i])
        data.append((choice_player, choice_cpu))
        state = make_state(choice_player, choice_cpu)

    data_players.append(data)

with open("./data/data_sample.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    # データを書き込む
    for i in range(n):
        row = np.array(data_players[i], dtype=np.int8)
        writer.writerow(row.flatten())

with open("./data/data_sample_param.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    # データを書き込む
    for i in range(n):
        row = np.array(true_mats[:,:,i], dtype=np.float32)
        writer.writerow(row.flatten())

  
print("data saved")