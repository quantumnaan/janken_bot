import numpy as np
from constatants import *

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
