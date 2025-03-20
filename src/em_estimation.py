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
    self.Sig = np.identity(NP)*10
    self.mu = np.zeros(NP)
    self.transmat = np.random.rand(3, NS, NP)*2-1
    self.transmat[2] = -self.transmat[0] - self.transmat[1]
    
  def load_data(self):
    assert os.path.exists(file_data), f"File {file_data} does not exist"
    with open(file_data, "rb") as f:
      loaded_data = pk.load(f)
    self.data = loaded_data
    print(f"data shape: {len(self.data)}")
    
  def param_estimation(self):
    for k in range(10):
      prev_mu = self.mu
      prev_Sig = self.Sig
      prev_transmat = self.transmat
      
      self.mu, self.Sig = self.Qestimation()
      print(f"{k+1} th mu / sig estimation finished:")
      print(self.mu)
      print(self.Sig)
      
      self.transmat = self.Testimaiton()
      print(f"{k+1} th transmat estimation finished:")
      print(self.transmat)
      input()
      
      if np.linalg.norm(prev_mu - self.mu, ord=1) < 1e-3 and \
        np.linalg.norm(prev_Sig.flatten() - self.Sig.flatten(), ord = 1) < 1e-3 and \
        np.linalg.norm(prev_transmat.flatten() - self.transmat.flatten(), ord = 1) < 1e-3:
          print("param estimation converged")
          break
        
        
  def MAP_ths(self):
    ths = []
    self.th_estimation.set_param(self.mu, self.Sig)
    self.th_estimation.set_transmat(self.transmat)
    for datum in self.data:
      ths.append(self.th_estimation.estimate(datum))
    return np.array(ths)
    
  def Qestimation(self) -> np.ndarray:
    ths = self.MAP_ths()
    
    n = ths.shape[0]
    mu_hat = np.mean(ths, axis=0).flatten()
    Sig_hat = np.zeros((NP, NP))
    
    for i in range(NP):
      for j in range(i+1):
        Sig_hat[i,j] = np.dot(ths[:,i].flatten() - mu_hat[i], ths[:,j].flatten() - mu_hat[j]) / (n-1)
        Sig_hat[j,i] = Sig_hat[i,j]
    
    return mu_hat, Sig_hat
  
  def Testimaiton(self) -> np.ndarray:
    ths = self.MAP_ths()
    
    likelihood = ca.MX(0)
    transmat2 = ca.MX.sym("transmat", 2*NS, NP)
    tp_mat2 = ca.reshape(transmat2, 2, NS*NP)
    transmat = ca.reshape(ca.vertcat(tp_mat2, -ca.sum1(tp_mat2)), 3*NS, NP)
    for i in range(len(self.data)):
      ja_prob = transmat @ ths[i]
      ja_prob = ca.reshape(ja_prob, 3, NS)
      ja_prob = ca.exp(ja_prob) + EPS
      ja_prob = ja_prob / ca.repmat(ca.sum1(ja_prob), 3, 1) # repmatは行方向に3行分複製
      x = self.data[i]
      for j in range(len(x)):
        likelihood -= ca.log(ja_prob[x[j,1],x[j,0]] + EPS)
        
    opts = {"print_time": False, "ipopt.print_level": 0}
    solver = ca.nlpsol("solver", "ipopt", {"x": ca.vec(transmat2), "f": likelihood}, opts)
    
    if "solT" not in dir(self):
      self.solT = solver(x0=np.ones(2 * NS * NP))
    else:
      warm_start = {"x0": self.solT["x"]}
      self.solT = solver(**warm_start)
    
    sol_mat2 = ca.reshape(self.solT["x"], 2, NS*NP)
    sol_mat = ca.vertcat(sol_mat2, -ca.sum1(sol_mat2)).full()
    sol_mat = sol_mat.reshape((3, NS, NP))
    mat_hat = sol_mat @ self.mu
    mat_hat = np.exp(mat_hat)
    mat_hat = mat_hat / np.sum(mat_hat, axis=0)
    print(mat_hat)
    input()
      
    return sol_mat
  
if __name__ == "__main__":
  param_estimation = ParamEstimation()
  param_estimation.load_data()
  param_estimation.param_estimation()
  print("\n mu:")
  print(param_estimation.mu)
  print("\n Sig:")
  print(param_estimation.Sig)
  print("\n transmat:")
  print(param_estimation.transmat)