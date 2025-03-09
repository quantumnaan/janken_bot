import casadi as ca
import numpy as np
import pickle as pk
import os

file_param = "./data/params.pkl"
file_mat = "./data/transmat.pkl"

class BayesEstimation:
  def __init__(self):
    assert os.path.exists(file_param), f"File {file_param} does not exist"
    assert os.path.exists(file_mat), f"File {file_mat} does not exist"
    self.file_param = file_param
    self.file_mat = file_mat
    
    self.Sig = ca.DM.zeros(3, 3)
    self.mu = ca.DM.zeros(3)
    self.Sig, self.mu = self._load_params()
    self.Sig_inv = ca.inv(self.Sig)
    self.transmat = self._load_transmat() # th -> janken_prob
    
    self.th = ca.MX.sym("th", 3)
    
    likelihood = self.log_pi(self.th)
    nlp = {"x": self.th, "f": likelihood}
    opts = {"print_time": False, "ipopt.print_level": 0}
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    self.sol = solver()
    

  def log_model(self, th:ca.MX, x:np.ndarray) -> ca.MX:
    ja_prob = self.transmat @ th
    ja_prob = ca.fmax(ja_prob, 1e-6)
    # ja_prob = ja_prob / ca.sum1(ja_prob)
    # TODO: モデルに応じた適切な確率の標準化を行う
    ret = ca.MX(0)
    for i in range(len(x)):
      ret -= ca.log(ja_prob[x[i]])
    return ret
  
  def log_pi(self, th:ca.MX) -> ca.MX:
    return 0.5* (th-self.mu).T @ self.Sig_inv @ (th-self.mu)
    
  def estimate(self, x: np.ndarray) -> np.ndarray:
    warm_start = {"x0": self.sol["x"]}
    
    if(x.shape[0] == 0):
      likelihood = self.log_pi(self.th)
    else:
      likelihood = self.log_model(self.th, x) + self.log_pi(self.th)
    
    nlp = {"x": self.th, "f": likelihood}
    
    opts = {"print_time": False, "ipopt.print_level": 0}
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    
    self.sol = solver(**warm_start)
    return self.transmat @ self.sol["x"]
  
  def _load_params(self):
    with open(self.file_param, "rb") as f:
      loaded_data = pk.load(f)
    return loaded_data["Sig"], loaded_data["mu"]
  
  def _load_transmat(self):
    with open(self.file_mat, "rb") as f:
      loaded_data = pk.load(f)
    return loaded_data["transmat"]
  
if __name__ == "__main__":
  be = BayesEstimation()
  x = np.array([0, 1, 2, 0, 0, 2])
  print(be.estimate(x))