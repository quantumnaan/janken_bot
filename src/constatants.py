import numpy as np
file_param = "./data/params.pkl"
file_mat = "./data/transmat.pkl"
file_data = "./data/data.pkl" # [人i:[ターンt:(人iが出した手, cpが出した手)]]
file_data_param = "./data/sample_data_param.pkl"
file_model = "./data/model.pkl"

# NP:パラメータ数，NS:状態数
NP = 4
NS = 9

EPS = 1e-12
lambda1_ = 1e-2 # em_estimationでのtransmat予測時の正則化
lambda2_ = 1e-2 # thのestimateでのエントロピー正則化
# lambda3_ = 0.02 # thのestimateでのエントロピー正則化
lambda4_ = 1e-1

np.random.seed(0) # 乱数固定
np.set_printoptions(precision=2, suppress=True) # printの精度設定

# 状態のインデックス対応 (i:(前に出した手，前の勝敗))
# 0: (グー, 負け), 1: (グー, あいこ), 2: (グー, 勝ち)
# 3: (チョキ, 負け), 4: (チョキ, あいこ), 5: (チョキ, 勝ち)
# 6: (パー, 負け), 7: (パー, あいこ), 8: (パー, 勝ち)