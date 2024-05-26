import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from matplotlib import colors
from tqdm import tqdm
import scipy.linalg as li
from scipy.optimize import minimize
plt.rcParams["font.size"] = 20

fig = plt.figure(figsize=(25, 6), facecolor="w", dpi=600)
label = ["NL-BFGS\n(NL-QUO)", "L-BFGS\n(L-QUO)", "Sim-QA\n(L-QUBO)", "Phy-QA\n(L-QUBO)"]
color_list = ["#0532FF", "#F72308", "orange", "magenta"]

ax = fig.add_subplot(1, 3, 1)
RMSE_FG = np.load('./data/RMSE_FG.npy')[0]
RMSE_NL = np.load('./data/RMSE_NL.npy')[0]
RMSE_L = np.load('./data/RMSE_L.npy')[0]
RMSE_SIM = np.load('./data/RMSE_SIM.npy')[0]
RMSE_PHY = np.load('./data/RMSE_PHY.npy')[0]
rmse_left = np.arange(1, 4+1)
rmse_height = np.array([RMSE_NL, RMSE_L, RMSE_SIM, RMSE_PHY])
ax.bar(rmse_left, rmse_height, tick_label=label, align="center", width=0.6, color=color_list)
ax.axhline(y=RMSE_FG, color="green", linestyle="--", linewidth=2.0)
ax.set_title("(a) Mean RMSE of Analysis")
ax.set_yticks(np.arange(0, 25+5, 5)/100)
ax.set_xlabel("Solver")
ax.set_ylabel("RMSE")
ax.set_ylim(0., 0.28)
ax.grid()
ax.set_axisbelow(True)

ax = fig.add_subplot(1, 3, 2)
RMSE_FG = np.load('./data/RMSE_FG.npy')[1]
RMSE_NL = np.load('./data/RMSE_NL.npy')[1]
RMSE_L = np.load('./data/RMSE_L.npy')[1]
RMSE_SIM = np.load('./data/RMSE_SIM.npy')[1]
RMSE_PHY = np.load('./data/RMSE_PHY.npy')[1]
rmse_left = np.arange(1, 4+1)
rmse_height = np.array([RMSE_NL, RMSE_L, RMSE_SIM, RMSE_PHY])
ax.bar(rmse_left, rmse_height, tick_label=label, align="center", width=0.6, color=color_list)
ax.axhline(y=RMSE_FG, color="green", linestyle="--", linewidth=2.0)
ax.set_title("(b) Mean RMSE of 2-day Forecast")
ax.set_xlabel("Solver")
ax.set_ylabel("RMSE")
ax.set_ylim(0., 0.5828)
ax.grid()
ax.set_axisbelow(True)

ax = fig.add_subplot(1, 3, 3)
TIME_NL = np.load('./data/TIME_NL.npy')
TIME_L = np.load('./data/TIME_L.npy')
TIME_SIM = np.load('./data/TIME_SIM.npy')
TIME_PHY = np.load('./data/TIME_PHY.npy')
time_left = np.arange(1, 4+1)
time_height = np.array([TIME_NL, TIME_L, TIME_SIM, TIME_PHY])
ax.bar(time_left, time_height, tick_label=label, align="center", width=0.6, color=color_list)
ax.set_title("(c) Mean Execution Time")
ax.set_xlabel("Solver")
ax.set_ylabel("Time [s]")
ax.set_yscale("log")
ax.grid()
ax.set_axisbelow(True)
plt.savefig("./fig/fig02")