from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

from constatants import *

def check_winlose(choice1:int, choice2:int):
  """
    choice1: プレイヤーの手
    choice2: CPの手
    return: 勝敗
  """
  return (choice2 - choice1 + 4) % 3 # 0:負け, 1:あいこ, 2:勝ち

class PlayerData:
  def __init__(self, data:np.ndarray):
    """
      data: [ターンtについて:[t番目の手, t番目のCPの手]]
    """
    self.data = data
    self.M = len(data) # ターン数M
    self.gu_sum = np.cumsum(data[:,0]==0)
    self.choki_sum = np.cumsum(data[:,0]==1)
    self.pa_sum = np.cumsum(data[:,0]==2)
    self.corr1st = self.calc_corr1st()
    if self.M > 2: self.corr2nd = self.calc_corr2nd()
    else : self.corr2nd = 0
    
  def calc_corr1st(self):
    """
      1つ前の手との相関係数
    """
    return np.corrcoef(self.data[1:,0], self.data[:-1,0])[0,1]
    
  def calc_corr2nd(self):
    """
      2つ前の手との相関係数
    """
    assert self.M > 2, "N should be greater than 2"
    return np.corrcoef(self.data[2:,0], self.data[:-2,0])[0,1]
  
  def state_vector(self, t:int):
    """
      ターン t+1 における手を予測するとき用の状態を返す
      state[1], corrをt依存するように変更する
    """
    assert t < self.M, f"t should be less than {self.M}"
    state = np.empty(NP)
    state[0] = self.data[t][0] # 直前の手
    state[1] = check_winlose(self.data[t-1][0],self.data[t-1][1]) # 直前の勝敗
    state[2] = self.gu_sum[t] # これまでのグーの数
    state[3] = self.choki_sum[t] # これまでのチョキの数
    state[4] = self.pa_sum[t] # これまでのパーの数
    state[5] = self.corr1st # 1つ前の手との相関係数
    state[6] = self.corr2nd # 2つ前の手との相関係数
    # 後は1つ前の相手(CP)の手との相関係数など
    
    return state
  
  def choice_ans(self, t:int):
    """
      ターン t における手を返す
    """
    assert t < self.M, f"t should be less than {self.M}"
    return self.data[t][0]
    
# とりあえず全データ(forall t)を使って学習
class MultiLogistic:
  def __init__(self, dataset:np.ndarray):
    self.dataset = dataset
    self.N = len(dataset) # 人の数N
    self.players = [PlayerData(data) for data in dataset]
    
    self.all_x = np.array([player.state_vector(player.M-2) for player in self.players])
    self.all_y = np.array([player.choice_ans(player.M-1) for player in self.players])
    self.train_x, self.test_x, self.train_y, self.test_y \
        = train_test_split(self.all_x, self.all_y, test_size=0.2, random_state=0)
    
    self.scaler = StandardScaler()
    self.train_x = self.scaler.fit_transform(self.train_x)
    self.test_x = self.scaler.transform(self.test_x)
    
  def train(self):
    assert len(self.train_x) == len(self.train_y), "train_x and train_y should have the same length"
    self.model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=200)
    self.model.fit(self.train_x, self.train_y)
  
  def predict(self, x:np.ndarray):
    x = self.scaler.transform(x)
    return self.model.predict(x)
  
  def evaluate(self):
    pred_y = self.predict(self.test_x)
    return accuracy_score(self.test_y, pred_y)

if __name__ == "__main__":
  
  def calc_choicemat(data:np.ndarray, model:MultiLogistic):
    """
      data: [ターンt:[[t番目の手, t番目のCPの手]]
    """
    choicemat = np.zeros((3, NS))
    for i in range(len(data)-1):
      win_lose = (data[i][1] - data[i][0] + 4) % 3
      statei = data[i][0]*3 + win_lose
      choicemat[data[i+1][0], statei] += 1
    choicemat = choicemat / np.sum(choicemat, axis=0, keepdims=True)
    return choicemat
    
  import pickle as pk
  file_data = "./data/sample_data.pkl"
  file_data_param = "./data/sample_data_param.pkl"
  
  with open(file_data, 'rb') as f:
    dataset = pk.load(f)
    
  model = MultiLogistic(dataset)
  print(np.unique(model.all_y))
  model.train()
  print(model.evaluate())
  
  with open(file_data_param, "rb") as f:
    loaded_data = pk.load(f)
    mu = loaded_data["mu"]
    Sig = loaded_data["Sig"]
    sample_data = loaded_data["sample_data"]
  
  print(f"予測した手の出し方:")
  print(calc_choicemat(dataset[0], model))
  print(f"正解の手の出し方:")
  print(sample_data[:,:,0])
  