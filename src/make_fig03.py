import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from matplotlib import colors
from tqdm import tqdm
plt.rcParams["font.size"] = 16


class Model() :
    def __init__(self, DT=0.05, DAY=100) :
        self.DT = DT
        self.DAY_STEP = int(0.2 / self.DT)
        self.DAY = DAY
        self.SIM_STEP = self.DAY * self.DAY_STEP
        self.SIM_IDX = self.SIM_STEP + 1
        self.F = 8.
        self.N = int(40)
        self.P = int(40)
    
    def lorenz96(self, x) :
        f = np.zeros((self.N))
        f[0] = (x[1] - x[self.N-2]) * x[self.N-1] - x[0] + self.F
        f[1] = (x[2] - x[self.N-1]) * x[0] - x[1] + self.F
        for n in range(2, self.N-1) : 
            f[n] = (x[n+1] - x[n-2]) * x[n-1] - x[n] + self.F
        f[self.N-1] = (x[0] - x[self.N-3]) * x[self.N-2] - x[self.N-1] + self.F
        return f
    
    def runge_kutta(self, x_old) :
        k1 = self.DT * self.lorenz96(x_old)
        k2 = self.DT * self.lorenz96(x_old + 0.5 * k1)
        k3 = self.DT * self.lorenz96(x_old + 0.5 * k2)
        k4 = self.DT * self.lorenz96(x_old + k3)
        x_new = x_old + (1. / 6.) * (k1 + 2. * k2 + 2. * k3 + k4)
        return x_new


class DataAssimilation(Model) :
    def __init__(self) :
        super().__init__()
        self.x_tru = np.zeros((self.SIM_IDX, self.N))
        self.x_tru[0, :] = np.load('./data/x_tru_init.npy')
        for t in range(self.SIM_STEP) :
            self.x_tru[t+1, :] = self.runge_kutta(self.x_tru[t, :])


class VariationalMethod(DataAssimilation) :
    def __init__(self, WINDOW_DAY):
        super().__init__()
        self.WINDOW_DAY = WINDOW_DAY
        self.WINDOW_DAY_STEP = self.WINDOW_DAY * self.DAY_STEP
        self.WINDOW_NUM = int(self.DAY / self.WINDOW_DAY)
        self.x_est = np.load('./data/x_est.npy')

    def plot_contour(self, CONTOUR_DAY, VARIABLE1, VARIABLE2, TOL) :
        plt.rcParams["font.size"] = 16
        VARIABLE1_MIN = 8.75
        VARIABLE1_MAX = 9.5
        VARIABLE2_MIN = 0.5
        VARIABLE2_MAX = 1.25
        tol = TOL
        cost_function_NL = np.load('./data/cost_function_NL.npy')
        cost_function_L = np.load('./data/cost_function_L.npy')
        cost_function_NL_min = np.min(cost_function_NL)
        cost_function_L_min = np.min(cost_function_L)
        levels_arr_NL = [cost_function_NL_min, cost_function_NL_min+0.1, cost_function_NL_min+0.2, cost_function_NL_min+0.4, cost_function_NL_min+1.0, cost_function_NL_min+2.0, cost_function_NL_min+4.0, cost_function_NL_min+10, cost_function_NL_min+20]
        levels_arr_L = [cost_function_L_min, cost_function_L_min+0.1, cost_function_L_min+0.2, cost_function_L_min+0.4, cost_function_L_min+1.0, cost_function_L_min+2.0, cost_function_L_min+4.0, cost_function_L_min+10, cost_function_L_min+20]
        w = int(CONTOUR_DAY / self.WINDOW_DAY)
        window_init_index = w * self.WINDOW_DAY_STEP
        x_opt_list_NL = np.load('./data/dx_opt_list_NL.npy') + self.x_est[window_init_index]
        x_opt_list_L = np.load('./data/dx_opt_list_L.npy') + self.x_est[window_init_index]
        x_SIM = np.load("./data/dx_opt_SIM.npy") + self.x_est[:-2*4:2*4]
        x_PHY = np.load("./data/dx_opt_PHY.npy") + self.x_est[:-2*4:2*4]
        fig = plt.figure(figsize=(10, 6), facecolor="w", dpi=600)
        X_0, X_1 = np.mgrid[VARIABLE1_MIN:VARIABLE1_MAX+tol:tol, VARIABLE2_MIN:VARIABLE2_MAX+tol:tol]
        cont_w = plt.contour(X_0, X_1, cost_function_NL, cmap="Blues_r", linewidths=1, linestyles="-", levels=levels_arr_NL, alpha=1.0)
        cont_wo = plt.contour(X_0, X_1, cost_function_L, cmap="Reds_r", linewidths=1, linestyles="-", levels=levels_arr_L, alpha=1.0)
        plt.scatter(self.x_tru[window_init_index, VARIABLE1], self.x_tru[window_init_index, VARIABLE2], label="True", c="black", s=150, alpha=1.0, marker="*", edgecolors="black", zorder=1)
        plt.scatter(self.x_est[window_init_index, VARIABLE1], self.x_est[window_init_index, VARIABLE2], label="FG", c="green", s=120, alpha=1.0, marker="o", edgecolors="black", zorder=2)
        plt.scatter(x_opt_list_NL[-1, VARIABLE1], x_opt_list_NL[-1, VARIABLE2], label="AN (NL-BFGS (NL-QUO))", c="#0532FF", s=120, alpha=1.0, marker="^", edgecolors="black", zorder=3)
        plt.scatter(x_opt_list_L[-1, VARIABLE1], x_opt_list_L[-1, VARIABLE2], label="AN (L-BFGS (L-QUO))", c="#F72308", s=120, alpha=1.0, marker="^", edgecolors="black", zorder=4)
        plt.scatter(x_SIM[w, VARIABLE1], x_SIM[w, VARIABLE2], label="AN (Sim-QA (L-QUBO))", c="orange", s=120, alpha=1.0, marker="^", edgecolors="black", zorder=5)
        plt.scatter(x_PHY[w, VARIABLE1], x_PHY[w, VARIABLE2], label="AN (Phy-QA (L-QUBO))", c="magenta", s=120, alpha=1.0, marker="^", edgecolors="black", zorder=6)
        for i in range(x_opt_list_NL.shape[0]-1) :
            plt.plot(x_opt_list_NL[i:i+2, VARIABLE1], x_opt_list_NL[i:i+2, VARIABLE2], linewidth=0.5, linestyle="-", color="#0532FF", marker="x", markersize=4, alpha=1.)
        for i in range(x_opt_list_L.shape[0]-1) :
            plt.plot(x_opt_list_L[i:i+2, VARIABLE1], x_opt_list_L[i:i+2, VARIABLE2], linewidth=0.5, linestyle="-", color="#F72308", marker="x", markersize=4, alpha=1.)
        plt.colorbar(cont_w)
        plt.colorbar(cont_wo)
        plt.title("Contour Lines of Cost Function")
        plt.xlabel("$x_2$")
        plt.ylabel("$x_3$")
        plt.xticks(np.arange(880, 940+10, 10)/100)
        plt.xlim(8.75, 9.5)
        plt.ylim(0.5, 1.25)
        plt.text(9.515, 0.42, "Cost Function\nw/ linearization", fontsize=12)
        plt.text(9.72, 0.42, "Cost Function\nw/o linearization", fontsize=12)
        l = plt.legend(fontsize=12)
        l.set_zorder(100)
        plt.savefig("./fig/fig03")


_4dvar = VariationalMethod(WINDOW_DAY=2)
_4dvar.plot_contour(CONTOUR_DAY=36, VARIABLE1=1, VARIABLE2=2, TOL=0.005)
