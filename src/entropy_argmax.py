import numpy as np
import matplotlib.pyplot as plt

p1s = []
p2s = []
N = 10

for i in range(N+1):
  for j in range(N+1-i):
    p1 = i / N
    p2 = j / N
    p1s.append(p1)
    p2s.append(p2)
  
# 3d
ax = plt.figure().subplots(subplot_kw={'projection': '3d'})
ax.plot_trisurf(p1s, p2s, [-np.log(max(p1,p2,(1-p1-p2))) for p1, p2 in zip(p1s, p2s)])
ax.plot_trisurf(p1s, p2s, [-(p1*np.log(p1) + p2*np.log(p2) + (1-p1-p2)*np.log(1-p1-p2)) for p1, p2 in zip(p1s, p2s)])
plt.show()