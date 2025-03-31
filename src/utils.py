import numpy as np
from constatants import *
import torch

def make_state(choice1, choice2): 
  """
  Args:
    choice1: 前に出した手
    choice2: 前にCPが(ランダムに)出した手
  Returns:
    この時の状態 cf. constatants.py
  """
  win_lose = (choice2 - choice1 + 4) %3 # 0:負け, 1:あいこ, 2:勝ち
  return (choice1*3 + win_lose)

def make_data_mat(data):
  """
  Args:
    data: 人i:[ターンt:(人iが出した手, cpが出した手)]
  Returns:
    data_mat: (3, NS) 各状態において各手を何回選んだかの行列
  """
  data_mat = torch.zeros((3, NS))
  m = data.shape[0]
  for t in range(m-1):
    state = make_state(data[t,0], data[t,1])
    data_mat[data[t+1,0]][state] += 1
  return data_mat