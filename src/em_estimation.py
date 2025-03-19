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
    self.Sig = ca.DX.zeros(NP, NP)
    self.mu = ca.DX.zeros(NP)
    self.transmat = ca.DX.zeros(3, NS, NP)
    
  def param_estimation(self):
    for _ in range(100):
      self.mu, self.Sig = self.Qestimation()
      self.transmat = self.Testimaiton()
    
  def MAP_ths(self):
    ths = []
    self.th_estimation.set_param(self.mu, self.Sig)
    self.th_estimation.set_transmat(self.transmat)
    for datum in self.data:
      ths.append(self.th_estimation.estimate(datum))
    return ths
    
  def Qestimation(self):
    ths = self.MAP_ths()
    
    likelihood = ca.MX(0)
    mu = ca.MX.sym("mu", NP)
    Sig = ca.MX.sym("Sig", NP, NP)
    Sig_inv = ca.inv(Sig)
    detSig = ca.det(Sig)
    for th in ths:
      likelihood += (th - mu).T @ Sig_inv @ (th - mu) + ca.log(detSig)
    solver = ca.nlpsol("solver", "ipopt", {"x": ca.vertcat(mu, Sig), "f": likelihood})
    self.sol = solver()
    return self.sol["x"]
  
  def Testimaiton(self):
    ths = self.MAP_ths()
    
    likelihood = ca.MX(0)
    transmat = ca.MX.sym("transmat", 3, NS, NP)
    for i in range(len(self.data)):
      ja_prob = transmat @ ths[i]
      ja_prob = ca.exp(ja_prob)
      ja_prob = ja_prob / ca.repmat(ca.sum(ja_prob, 0), 3, 1) # repmatは行方向に3行分複製
      x = self.data[i]
      for j in range(len(x)):
        likelihood += - ca.log(ja_prob[x[j][0]][x[i][1]])
    solver = ca.nlpsol("solver", "ipopt", {"x": transmat, "f": likelihood})
    self.sol = solver()
    return self.sol["x"]