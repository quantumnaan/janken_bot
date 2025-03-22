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
    self.data_param = {}
    self.Sig = np.identity(NP)*2
    self.mu = np.zeros(NP)
    self.transmat = np.random.randn(3, NS, NP)
    
  def load_data(self):
    assert os.path.exists(file_data), f"File {file_data} does not exist"
    with open(file_data, "rb") as f:
      loaded_data = pk.load(f)
    self.data = loaded_data
    with open(file_data_param, "rb") as f:
      loaded_data = pk.load(f)
      self.data_param = loaded_data
      self.mu = self.data_param["mu"]
      self.Sig = self.data_param["Sig"]
      self.transmat = self.data_param["transmat"]
    print(self.data_param["mu"])
    print(self.data_param["Sig"])
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
      
      # for i in range(3): self.transmat = self.Testimaiton()
      self.transmat = self.Testimaiton()
      print(f"{k+1} th transmat estimation finished:")
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
    
    # ths += np.random.multivariate_normal(np.zeros(NP), 0.1 * np.identity(NP), size=n)
    
    for i in range(NP):
      for j in range(i+1):
        Sig_hat[i,j] = np.dot(ths[:,i].flatten() - mu_hat[i], ths[:,j].flatten() - mu_hat[j]) / (n-1)
        Sig_hat[j,i] = Sig_hat[i,j]
    
    return mu_hat, Sig_hat
  
  def Testimaiton(self) -> np.ndarray:
    ths = self.MAP_ths()
    
    likelihood = ca.MX(0)
    transmat = ca.MX.sym("transmat", 3*NS, NP)
    # transmat2 = ca.MX.sym("transmat", 2*NS, NP)
    # tp_mat2 = ca.reshape(transmat2, 2, NS*NP)
    # transmat = ca.reshape(ca.vertcat(tp_mat2, -ca.sum1(tp_mat2)), 3*NS, NP)
    for i in range(len(self.data)):
      ja_prob1 = transmat @ ths[i]
      ja_prob2 = ca.reshape(ja_prob1, 3, NS)
      ja_prob3 = ca.exp(ja_prob2)
      ja_prob = ja_prob3 / ca.repmat(ca.sum1(ja_prob3), 3, 1) # repmatは行方向に3行分複製

      x = self.data[i]
      for j in range(len(x)):
        likelihood = likelihood - ca.log(ja_prob[x[j,1],x[j,0]] + EPS)
      
      # エントロピー正則化
      for j in range(NS):
        likelihood = likelihood + lambda2_ * ca.sum1((ja_prob[:,j]+EPS) * ca.log(ja_prob[:,j]))
    
    # likelihood += likelihood + lambda1_ * ca.norm_2(ca.sum1(ca.reshape(transmat, 3, NS*NP))) # 冗長な分はこれで正則化されるはず
    likelihood = likelihood + lambda1_ * ca.norm_2(ca.vec(transmat)) # 冗長な分はこれで正則化されるはず
        
    opts = {"print_time": False, "ipopt.print_level": 0}
    # gs = [ca.sum1(ca.reshape(transmat, 3, NS*NP))]
    # lbg = [np.zeros(NS*NP)]
    # ubg = [np.zeros(NS*NP)]
    # gs = ca.vertcat(*gs)
    # lbg = ca.vertcat(*lbg)
    # ubg = ca.vertcat(*ubg)
    solver = ca.nlpsol("solver", "ipopt", {"x": ca.vec(transmat), "f": likelihood}, opts)
    
    if "solT" not in dir(self):
      self.solT = solver(x0=np.random.randn(3 * NS * NP))
    else:
      warm_start = {"x0": self.solT["x"]}
      self.solT = solver(**warm_start)
    
    # sol_mat2 = ca.reshape(self.solT["x"], 2, NS*NP)
    # sol_mat = ca.vertcat(sol_mat2, -ca.sum1(sol_mat2)).full()
    sol_mat = self.solT["x"].full()
    sol_mat = sol_mat.reshape((3, NS, NP))
    # mat_hat = sol_mat @ self.mu
    # mat_hat = np.exp(mat_hat)
    # mat_hat = mat_hat / np.sum(mat_hat, axis=0)
    # print(mat_hat)
    # input()
      
    return sol_mat
  
  def Pmat_estimation(self, x:np.ndarray) -> np.ndarray:
    self.th_estimation.set_param(self.mu, self.Sig)
    self.th_estimation.set_transmat(self.transmat)
    th = self.th_estimation.estimate(x, MAP=True)
    Pmat = self.transmat @ th
    Pmat = np.exp(Pmat) / np.sum(np.exp(Pmat), axis=0)
    return Pmat
  
if __name__ == "__main__":
  param_estimation = ParamEstimation()
  param_estimation.load_data()
  param_estimation.param_estimation()
  print("\n mu:")
  print(param_estimation.mu)
  print("\n Sig:")
  print(param_estimation.Sig)
  print("\n true mu:")
  print(param_estimation.data_param["mu"])
  print("\n true Sig:")
  print(param_estimation.data_param["Sig"])
  print("\n ex. 人 0 の手の出し方(真の値):")
  print(param_estimation.data_param["sample_data"][:,:,1])
  print("\n ex. 人 0 の手の出し方(予測値):")
  print(param_estimation.Pmat_estimation(param_estimation.data[1]))
  
  # print("\n transmat:")
  # print(param_estimation.transmat)