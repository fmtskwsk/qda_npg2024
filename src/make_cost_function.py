import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
import scipy.linalg as li
import scipy.stats as st
from scipy.optimize import minimize


class Model() :
    def __init__(self, DT=0.05, DAY=100, TLM_DELTA=1e-5) :
        SEED = 68
        np.random.seed(seed=SEED)
        self.DT = DT
        self.DAY_STEP = int(0.2 / self.DT)
        self.DAY = DAY
        self.SIM_STEP = self.DAY * self.DAY_STEP
        self.SIM_IDX = self.SIM_STEP + 1
        self.F = 8.
        self.N = int(40)
        self.TLM_DELTA = TLM_DELTA
    
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

    def tangent_linear_model(self, x) :
        M_jacobian = np.zeros((self.N, self.N))
        for n in range(self.N):
            e = np.zeros((self.N))
            e[n] = 1.
            M_jacobian[:, n] = (self.runge_kutta(x + self.TLM_DELTA * e) - self.runge_kutta(x)) / self.TLM_DELTA
        return M_jacobian


class DataAssimilation(Model) :
    def __init__(self) :
        super().__init__()
        self.P = int(40)
        self.RANDOM_OBS_MEAN = 0.
        self.RANDOM_OBS_STD = 1.
        self.H = np.zeros((self.P, self.N))
        for i in range(self.P) :
            self.H[i, i] = 1.
        self.R = np.identity((self.P)) * (self.RANDOM_OBS_STD**2)
        self.x_tru = np.zeros((self.SIM_IDX, self.N))
        self.y_o = np.zeros((self.SIM_IDX, self.N))
        self.x_tru[0, :] = np.load('./data/x_tru_init.npy')
        for t in range(self.SIM_STEP) :
            self.x_tru[t+1, :] = self.runge_kutta(self.x_tru[t, :])
        self.y_o = self.x_tru + np.random.normal(self.RANDOM_OBS_MEAN, self.RANDOM_OBS_STD, self.x_tru.shape)


class VariationalMethod(DataAssimilation) :
    def __init__(self, WINDOW_DAY, b_ii):
        super().__init__()
        self.WINDOW_DAY = WINDOW_DAY
        self.WINDOW_DAY_STEP = self.WINDOW_DAY * self.DAY_STEP
        self.WINDOW_NUM = int(self.DAY / self.WINDOW_DAY)
        self.B = np.identity((self.N)) * b_ii
        self.x_a = np.zeros((self.SIM_IDX, self.N))
        self.x_b = np.zeros((self.SIM_IDX, self.N))
        self.dx_b = np.zeros((self.SIM_IDX, self.N))
        self.x_b[0, :] = np.load('./data/x_b_init.npy')
        self.x_est = np.load('./data/x_est.npy')
    
    def make_data(self, VARIABLE1, VARIABLE2, CONTOUR_DAY, TOL) :
        def cost_function_NL(dx_0_1_opt, window_init_index, window_next_index) :
            dx_opt = np.zeros((self.N))
            dx_opt[VARIABLE1:VARIABLE2+1] = dx_0_1_opt
            dx_tru = self.x_tru[window_init_index, :] - self.x_b[window_init_index, :]
            dx_opt[:VARIABLE1] = dx_tru[:VARIABLE1]
            dx_opt[VARIABLE2+1:] = dx_tru[VARIABLE2+1:]
            J_b = dx_opt.T @ LA.inv(self.B) @ dx_opt
            J_o = 0.
            x_w = np.zeros((self.SIM_IDX+1, self.N))
            x_w[window_init_index, :] = self.x_b[window_init_index, :] + dx_opt
            x_w[window_init_index+1, :] = self.runge_kutta(x_w[window_init_index, :])
            dx_w = np.zeros((self.SIM_IDX+1, self.N))
            dx_w[window_init_index, :] = dx_opt
            dx_w[window_init_index+1, :] = x_w[window_init_index+1, :] - self.x_b[window_init_index+1, :]
            for i, t in enumerate(range(window_init_index+1, window_next_index+1)) :
                d = self.d_1L_NL[i]
                Z = self.H @ dx_w[t, :] - d
                J_o += Z.T @ LA.inv(self.R) @ Z
                if t < window_next_index :
                    x_w[t+1, :] = self.runge_kutta(x_w[t, :])
                    dx_w[t+1, :] = x_w[t+1, :] - self.x_b[t+1, :]
            J = J_b + J_o
            return J
        
        def cost_function_L(dx_0_1_opt) :
            dx_opt = np.zeros((self.N))
            dx_opt[VARIABLE1:VARIABLE2+1] = dx_0_1_opt
            dx_tru = self.x_tru[window_init_index, :] - self.x_b[window_init_index, :]
            dx_opt[:VARIABLE1] = dx_tru[:VARIABLE1]
            dx_opt[VARIABLE2+1:] = dx_tru[VARIABLE2+1:]
            J_b = dx_opt.T @ LA.inv(self.B) @ dx_opt
            J_o = 0.
            for i in range(self.WINDOW_DAY_STEP) :
                d = self.d_1L_L[i]
                M_t_0 = self.M_1L_0_L[i, :, :]
                Z = self.H @ M_t_0 @ dx_opt - d
                J_o += Z.T @ LA.inv(self.R) @ Z
            J = J_b + J_o
            return J
        
        w = int(CONTOUR_DAY / self.WINDOW_DAY)
        window_init_index = w * self.WINDOW_DAY_STEP       
        window_next_index = (w + 1) * self.WINDOW_DAY_STEP 
        tol = TOL
        VARIABLE1_MIN = 8.75
        VARIABLE1_MAX = 9.5
        VARIABLE2_MIN = 0.5
        VARIABLE2_MAX = 1.25
        X_0_list = np.arange(VARIABLE1_MIN, VARIABLE1_MAX+tol, tol)
        X_1_list = np.arange(VARIABLE2_MIN, VARIABLE2_MAX+tol, tol)
        self.cost_function_NL = np.zeros((len(X_0_list), len(X_1_list)))
        self.cost_function_L = np.zeros((len(X_0_list), len(X_1_list)))
        self.x_b[window_init_index, :] = self.x_est[window_init_index, :]
        self.d_1L_NL = self.generate_window_data_NL(window_init_index, window_next_index)
        self.M_1L_0_L, self.d_1L_L = self.generate_window_data_L(window_init_index, window_next_index)
        for i, x_0 in enumerate(X_0_list) : 
            for j, x_1 in enumerate(X_1_list) :
                dx_0_1_opt = np.array([x_0, x_1]) - self.x_b[window_init_index, VARIABLE1:VARIABLE2+1]
                self.cost_function_NL[i, j] = cost_function_NL(dx_0_1_opt, window_init_index, window_next_index)
                self.cost_function_L[i, j] = cost_function_L(dx_0_1_opt)
        np.save('./data/cost_function_NL', self.cost_function_NL)
        np.save('./data/cost_function_L', self.cost_function_L)

    def generate_window_data_NL(self, window_init_index, window_next_index) : 
        d_1L = np.zeros((self.WINDOW_DAY_STEP, self.P))
        self.x_b[window_init_index+1, :] = self.runge_kutta(self.x_b[window_init_index, :])
        for i, t in enumerate(range(window_init_index+1, window_next_index+1), 1) : 
            d = self.y_o[t, :] - self.H @ self.x_b[t, :]
            d_1L[i-1] = d
            if t < window_next_index :
                self.x_b[t+1, :] = self.runge_kutta(self.x_b[t, :])
        return d_1L

    def generate_window_data_L(self, window_init_index, window_next_index) : 
        M_1L_0 = np.zeros((self.WINDOW_DAY_STEP, self.N, self.N))
        d_1L = np.zeros((self.WINDOW_DAY_STEP, self.P))
        x_b = np.zeros((self.SIM_IDX+1, self.N))
        x_b[window_init_index, :] = self.x_b[window_init_index, :]
        x_b[window_init_index+1, :] = self.runge_kutta(x_b[window_init_index, :])
        M_t = self.tangent_linear_model(x_b[window_init_index, :])
        M_t_0 = M_t
        M_1L_0[0, :, :] = M_t_0
        for i, t in enumerate(range(window_init_index+1, window_next_index+1), 1) : 
            d = self.y_o[t, :] - self.H @ x_b[t, :]
            d_1L[i-1] = d
            if t < window_next_index :
                M_t = self.tangent_linear_model(x_b[t, :])
                M_t_0 = M_t @ M_t_0
                M_1L_0[i, :, :] = M_t_0
                x_b[t+1, :] = self.runge_kutta(x_b[t, :])
        return M_1L_0, d_1L


_4dvar = VariationalMethod(WINDOW_DAY=2, b_ii=0.15)
_4dvar.make_data(VARIABLE1=1, VARIABLE2=2, CONTOUR_DAY=36, TOL=0.005)

