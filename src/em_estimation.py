import casadi as ca
import numpy as np
import pickle as pk
import os
from bayes_estimation import BayesEstimation

from constatants import *

# class GMMEstimation:
#   def __init__(self):
    

class ParamEstimation:
  def __init__(self):
    self.th_estimation = BayesEstimation()
    self.data = []
    self.Sig = np.identity(NP)
    self.mu = np.zeros(NP)
    self.transmat = np.ones((3, NS, NP))
    
  def load_data(self):
    assert os.path.exists(file_data), f"File {file_data} does not exist"
    with open(file_data, "rb") as f:
      loaded_data = pk.load(f)
    self.data = loaded_data
    print(f"data shape: {len(self.data)}")
    
  def param_estimation(self):
    for _ in range(100):
      prev_mu = self.mu
      prev_Sig = self.Sig
      prev_transmat = self.transmat
      ans_vec = self.Qestimation()
      self.mu = ans_vec[:NP].full()
      self.Sig = ans_vec[NP:].full().reshape((NP, NP))
      self.transmat = self.Testimaiton().full().reshape((3, NS, NP))
      if np.linalg.norm(prev_mu - self.mu, ord=1) < 1e-3 and \
        np.linalg.norm(prev_Sig - self.Sig, ord = 1) < 1e-3 and \
        np.linalg.norm(prev_transmat.flatten() - self.transmat.flatten(), ord = 1) < 1e-3:
          print("param estimation converged")
          break
    
  def MAP_ths(self):
    ths = []
    self.th_estimation.set_param(self.mu, self.Sig)
    self.th_estimation.set_transmat(self.transmat)
    for datum in self.data:
      ths.append(self.th_estimation.estimate(datum))
    return ths
    
  def Qestimation(self):
    ths = self.MAP_ths()
    print(ths)
    
    likelihood = ca.MX(0)
    mu = ca.MX.sym("mu", NP)
    Sig = ca.MX.sym("Sig", NP, NP)
    Sig_inv = ca.inv(Sig)
    detSig = ca.det(Sig)
    for th in ths:
      likelihood += (th - mu).T @ Sig_inv @ (th - mu) + ca.log(detSig)
      
    # 最適化についてprintしない
    opts = {"print_time": False, "ipopt.print_level": 0}
    solver = ca.nlpsol("solver", "ipopt", {"x": ca.vertcat(mu, ca.vec(Sig)), "f": likelihood}, opts)
    if "solQ" not in dir(self):
      self.solQ = solver(x0=ca.vertcat(self.mu, ca.vec(self.Sig)))
    else:
      warm_start = {"x0": self.solQ["x"]}
      self.solQ = solver(**warm_start)
    return self.solQ["x"]
  
  def Testimaiton(self):
    ths = self.MAP_ths()
    
    likelihood = ca.MX(0)
    transmat = ca.MX.sym("transmat", 3*NS, NP)
    for i in range(len(self.data)):
      ja_prob = transmat @ ths[i]
      ja_prob = ca.reshape(ja_prob, 3, NS)
      ja_prob = ca.exp(ja_prob)
      ja_prob = ja_prob / ca.repmat(ca.sum1(ja_prob), 3, 1) # repmatは行方向に3行分複製
      x = self.data[i]
      for j in range(len(x)):
        likelihood -= ca.log(ja_prob[x[j][1],x[j][0]])
        
    opts = {"print_time": False, "ipopt.print_level": 0}
    solver = ca.nlpsol("solver", "ipopt", {"x": ca.vec(transmat), "f": likelihood}, opts)
    
    if "solT" not in dir(self):
      self.solT = solver(x0=np.ones(3 * NS * NP))
    else:
      warm_start = {"x0": self.solT["x"]}
      self.solT = solver(**warm_start)
      
    return self.solT["x"]
  
if __name__ == "__main__":
  param_estimation = ParamEstimation()
  param_estimation.load_data()
  param_estimation.param_estimation()
  print(param_estimation.mu)
  print(param_estimation.Sig)
  print(param_estimation.transmat)