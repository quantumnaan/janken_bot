import casadi as ca
import numpy as np
import pickle as pk
import os

from constatants import *

class BayesEstimation:
  def __init__(self):
    assert os.path.exists(file_param), f"File {file_param} does not exist"
    assert os.path.exists(file_mat), f"File {file_mat} does not exist"
    self.file_param = file_param
    self.file_mat = file_mat
    
    self.Sig = np.identity(NP)
    self.mu = np.zeros(NP)
    self.transmat = np.ones((3, NS, NP))
    
  def set_param(self, mu, Sigma):
    self.mu = mu
    self.Sig = Sigma
    self.Sig_inv = np.linalg.inv(self.Sig)
    
  def set_transmat(self, transmat):
    self.transmat = transmat    

  def log_model(self, th:ca.MX, x:np.ndarray) -> ca.MX:
    # x[i] = [i番目の状態, 直後に出した手]
    tp_transmat = self.transmat.reshape(-1, NP)
    ja_prob = tp_transmat @ th
    ja_prob = ca.reshape(ja_prob, 3, NS)
    ja_prob = ca.exp(ja_prob)
    ja_prob = ja_prob / ca.repmat(ca.sum1(ja_prob), 3, 1) # repmatは行方向に3行分複製
    
    ret = ca.MX(0)
    for i in range(len(x)):
      ret -= ca.log(ja_prob[x[i,1],x[i,0]] + EPS)
    return ret
  
  def log_pi(self, th:ca.MX) -> ca.MX:
    return 0.5* (th-self.mu).T @ self.Sig_inv @ (th-self.mu)
    
  def estimate(self, x: np.ndarray) -> np.ndarray:
    th = ca.MX.sym("th", NP)
    likelihood = self.log_model(th, x) + self.log_pi(th)
    # 最適化についてprintしない
    opts = {"print_time": False, "ipopt.print_level": 0}
    
    solver = ca.nlpsol("solver", "ipopt", {"x": th, "f": likelihood}, opts)
    if "sol" not in dir(self):
      self.sol = solver(x0=np.ones(NP))
    else:
      warm_start = {"x0": self.sol["x"]}
      self.sol = solver(**warm_start)
    return self.sol["x"].full()
  
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
  x = np.array([[0, 1], [2, 0], [0, 2]])
  print(be.estimate(x))