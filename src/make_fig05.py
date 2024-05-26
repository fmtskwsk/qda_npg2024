import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from matplotlib import colors
plt.rcParams["font.size"] = 20
plt.rcParams['axes.axisbelow'] = True

fig = plt.figure(figsize=(8, 6), facecolor="w", dpi=600)
RMSE_FG = np.load('./data/RMSE_FG.npy')[0]
RMSE_SIM = np.load("./data/RMSE_SCALING_FACTOR_SIM.npy")
RMSE_PHY = np.load("./data/RMSE_SCALING_FACTOR_PHY.npy")
SCALING_FACTOR = np.array([5, 10, 20, 50, 100, 200, 500])
plt.axhline(y=RMSE_FG, color="green", linestyle="--", linewidth=1.5)
plt.plot(SCALING_FACTOR, RMSE_SIM, label="Sim-QA (L-QUBO)", color="orange", linewidth=1.5, marker="o", markersize=10)
plt.plot(SCALING_FACTOR, RMSE_PHY, label="Phy-QA (L-QUBO)", color="magenta", linewidth=1.5, marker="o", markersize=10)
plt.title(r"Sensitivity of RMSEs to the Scaling Factor $\alpha$")
plt.xlabel(r"Scaling Factor $\alpha$")
plt.ylabel("RMSE")
plt.xscale("log")
plt.xticks([5, 10, 20, 50, 100, 200, 500], ["5", "10", "20", "50", "100", "200", "500"])
plt.legend()
plt.grid()
plt.savefig("./fig/fig05")
