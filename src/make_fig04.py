import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from matplotlib import colors
plt.rcParams["font.size"] = 20

fig = plt.figure(figsize=(18, 6), facecolor="w", dpi=600)
label = ["1", "2", "5", "10", "20", "50", "100", "150", "200"]
# RMSE
ax_rmse = fig.add_subplot(1,2,1)
RMSE_PHY = np.load("./data/RMSE_NUM_READS_PHY.npy")
STD_PHY = np.load("./data/STD_NUM_READS_PHY.npy")
rmse_left = np.arange(1, 9+1)
rmse_height = RMSE_PHY
ax_rmse.bar(rmse_left, rmse_height, tick_label=label, align="center", width=0.6, color="magenta", yerr=STD_PHY, capsize=6)
ax_rmse.set_title("(a) Mean RMSE")
ax_rmse.set_yticks(np.arange(0, 230+5, 5)/1000)
ax_rmse.set_xlabel("num_reads")
ax_rmse.set_ylabel("RMSE")
ax_rmse.set_ylim(0.20, 0.23)
ax_rmse.grid()
ax_rmse.set_axisbelow(True)
# Time
ax_time = fig.add_subplot(1,2,2)
TIME_PHY = np.load("./data/TIME_NUM_READS_PHY.npy")
X = np.array([1, 2, 5, 10, 20, 50, 100, 150, 200])
Y = TIME_PHY
ax_time.plot(X, Y, color="magenta", linewidth=1.8, marker="o", markersize=8)
ax_time.set_title("(b) Mean Execution Time")
ax_time.set_xlabel("num_reads")
ax_time.set_ylabel("Time [s]")
ax_time.set_yticks(np.arange(0, 80+10, 10)/1000)
ax_time.grid()
ax_time.set_axisbelow(True)
plt.savefig("./fig/fig04")
