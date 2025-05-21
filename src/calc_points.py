import csv
import numpy as np

# 基本はmain_appで追加されていくが，
# リセットしたいときなどはこっからpointsをdata.csvから計算する

file_points = "./data/points.csv"
file_data = "./data/data.csv"

with open(file_data, 'r') as f:
  reader = csv.reader(f)
  data = []
  for row in reader:
    data.append(row)
data = np.array(data)

with open(file_points, 'w', newline='') as f:
    writer = csv.writer(f)
    for i in range(len(data)):
        nums = 0
        point = 0.
        for j in range(0, len(data[i]), 2):
            nums += 1
            if (int(data[i][j+1]) - int(data[i][j]) + 4)%3 == 2:
                point += 1
            elif (int(data[i][j+1]) - int(data[i][j]) + 4)%3 == 0:
                point -= 1
            else:
                point += 0
        point = point / nums
        writer.writerow([point])