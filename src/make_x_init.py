import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
import scipy.linalg as li
import scipy.stats as st
from scipy.optimize import minimize


class Model() :
    def __init__(self, DT=0.05, DAY=365, TLM_DELTA=1e-5) :
        SEED = 68
        np.random.seed(seed=SEED)
        self.DT = DT
        self.DAY_STEP = int(0.2 / self.DT)
        self.DAY = DAY
        self.SIM_STEP = self.DAY * self.DAY_STEP
        self.SIM_IDX = self.SIM_STEP + 1
        self.YEAR_STEP = 364 * self.DAY_STEP
        self.SPIN_UP_STEP = self.YEAR_STEP
        self.NATURE_RUN_STEP = self.YEAR_STEP * 3
        self.NATURE_RUN_IDX = self.NATURE_RUN_STEP + 1
        self.F = 8.
        self.N = int(40)
        self.TLM_DELTA = TLM_DELTA
        self.x_nr = np.ones((self.NATURE_RUN_IDX, self.N)) * float(self.F)
        self.x_nr[:, int(self.N/2)] *= 1.001
        for t in range(self.NATURE_RUN_STEP) :
            self.x_nr[t+1, :] = self.runge_kutta(self.x_nr[t, :])
    
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
    def __init__(self):
        super().__init__()
        self.P = int(40)
        self.RANDOM_OBS_MEAN = 0.
        self.RANDOM_OBS_STD = 1.
        self.H = np.zeros((self.P, self.N))
        for i in range(self.P) :
            self.H[i, i] = 1.
        self.R = np.identity((self.P)) * (self.RANDOM_OBS_STD**2)
        self.x_tru = self.x_nr[self.SPIN_UP_STEP:self.SPIN_UP_STEP+self.SIM_IDX, :]
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
        self.x_b[0, :] = self.x_nr[np.random.randint(self.SPIN_UP_STEP-10, self.SPIN_UP_STEP)]
    
    def four_d_var_increment(self) :
        def cost_function(dx_opt, window_init_index, window_next_index) :
            J_b = dx_opt.T @ LA.inv(self.B) @ dx_opt
            J_o = 0.
            x_w = np.zeros((self.SIM_IDX+1, self.N))
            x_w[window_init_index, :] = self.x_b[window_init_index, :] + dx_opt
            x_w[window_init_index+1, :] = self.runge_kutta(x_w[window_init_index, :])
            dx_w = np.zeros((self.SIM_IDX+1, self.N))
            dx_w[window_init_index, :] = dx_opt
            dx_w[window_init_index+1, :] = x_w[window_init_index+1, :] - self.x_b[window_init_index+1, :]
            for i, t in enumerate(range(window_init_index+1, window_next_index+1)) :
                d = self.d_1L[i]
                Z = self.H @ dx_w[t, :] - d
                J_o += Z.T @ LA.inv(self.R) @ Z
                if t < window_next_index :
                    x_w[t+1, :] = self.runge_kutta(x_w[t, :])
                    dx_w[t+1, :] = x_w[t+1, :] - self.x_b[t+1, :]
            J = J_b + J_o
            return J
        
        def jacobian(dx_opt, window_init_index, window_next_index) :
            dJ_b = LA.inv(self.B) @ dx_opt
            dJ_o = 0.
            x_w = np.zeros((self.SIM_IDX+1, self.N))
            x_w[window_init_index, :] = self.x_b[window_init_index, :] + dx_opt
            x_w[window_init_index+1, :] = self.runge_kutta(x_w[window_init_index, :])
            M_t = self.tangent_linear_model(x_w[window_init_index, :])
            M_t_0 = M_t
            dx_w = np.zeros((self.SIM_IDX+1, self.N))
            dx_w[window_init_index, :] = dx_opt
            dx_w[window_init_index+1, :] = x_w[window_init_index+1, :] - self.x_b[window_init_index+1, :]
            for i, t in enumerate(range(window_init_index+1, window_next_index+1)) :
                d = self.d_1L[i]
                Z = self.H @ dx_w[t, :] - d
                dJ_o += M_t_0.T @ self.H.T @ LA.inv(self.R) @ Z
                if t < window_next_index :
                    x_w[t+1, :] = self.runge_kutta(x_w[t, :])
                    M_t = self.tangent_linear_model(x_w[t, :])
                    M_t_0 = M_t @ M_t_0
                    dx_w[t+1, :] = x_w[t+1, :] - self.x_b[t+1, :]
            dJ = dJ_b + dJ_o
            return dJ

        for w in tqdm(range(self.WINDOW_NUM)) :                       
            window_init_index = w * self.WINDOW_DAY_STEP              
            window_next_index = (w + 1) * self.WINDOW_DAY_STEP        
            dx_opt = np.zeros((self.N))
            self.d_1L = self.generate_window_data(window_init_index, window_next_index)
            optimal_solution = minimize(cost_function, dx_opt, args=(window_init_index, window_next_index), jac=jacobian, method="BFGS")
            self.dx_b[window_init_index, :] = optimal_solution.x
            self.x_a[window_init_index, :] = self.x_b[window_init_index, :] + self.dx_b[window_init_index, :]
            for t in range(window_init_index, window_next_index) :
                self.x_a[t+1, :] = self.runge_kutta(self.x_a[t, :])
            self.x_b[window_next_index, :] = self.x_a[window_next_index, :]

    def generate_window_data(self, window_init_index, window_next_index) : 
        d_1L = np.zeros((self.WINDOW_DAY_STEP, self.P))
        self.x_b[window_init_index+1, :] = self.runge_kutta(self.x_b[window_init_index, :])
        for i, t in enumerate(range(window_init_index+1, window_next_index+1), 1) : 
            d = self.y_o[t, :] - self.H @ self.x_b[t, :]
            d_1L[i-1] = d
            if t < window_next_index :
                self.x_b[t+1, :] = self.runge_kutta(self.x_b[t, :])
        return d_1L
    
    def make_data(self) :
        x_tru_init = self.x_tru[1448]   # index of 1448 : 362 day
        x_b_init = self.x_b[1448]
        np.save('./data/x_tru_init', x_tru_init)
        np.save('./data/x_b_init', x_b_init)


_4dvar = VariationalMethod(WINDOW_DAY=2, b_ii=0.15)
_4dvar.four_d_var_increment()
_4dvar.make_data()