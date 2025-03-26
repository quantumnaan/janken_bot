import numpy as np
file_param = "./data/params.pkl"
file_mat = "./data/transmat.pkl"
file_data = "./data/sample_data.pkl" # [人i:[(t番目の状態, 直後に出した手)]]
file_data_param = "./data/sample_data_param.pkl"

# NP:パラメータ数
NP = 7
NS = 9 # データ生成時の状態数
EPS = 1e-12

np.random.seed(0) # 乱数固定
np.set_printoptions(precision=2)

# 状態のインデックス対応 (i:(前に出した手，前の勝敗))
# 0: (グー, 負け), 1: (グー, あいこ), 2: (グー, 勝ち)
# 3: (チョキ, 負け), 4: (チョキ, あいこ), 5: (チョキ, 勝ち)
# 6: (パー, 負け), 7: (パー, あいこ), 8: (パー, 勝ち)