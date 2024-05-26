import subprocess
import numpy as np


class Model() :
    def __init__(self, DT=0.05, DAY=100) :
        self.DT = DT
        self.DAY_STEP = int(0.2 / self.DT)
        self.DAY = DAY
        self.SIM_STEP = self.DAY * self.DAY_STEP
        self.SIM_IDX = self.SIM_STEP + 1
        self.F = 8.
        self.N = int(40)

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


class Annealing(Model) :
    def __init__(self, WINDOW_DAY=2) :
        super().__init__()
        self.WINDOW_DAY = WINDOW_DAY
        self.WINDOW_DAY_STEP = self.WINDOW_DAY * self.DAY_STEP
        self.WINDOW_NUM = int(self.DAY / self.WINDOW_DAY)

    def quantum_annealing(self, BIT_NUM=4, SCALING_FACTOR=20, TIMEOUT=100) :
        command_list = ["/usr/bin/python3", "./src/QA_SIM.py", str(BIT_NUM), str(SCALING_FACTOR), str(TIMEOUT)]
        subprocess.run(command_list)
    
    def calculate_rmse(self) :
        dx_opt = np.load("./data/dx_opt_SIM.npy")
        x_est = np.load('./data/x_est.npy')
        x_opt = dx_opt + x_est[:-self.WINDOW_DAY_STEP:self.WINDOW_DAY_STEP]
        fcst_leng = (0, 2)
        rmse_list = []
        rmse_avg_list = np.zeros((len(fcst_leng)))
        MAX_SIM_STEP = self.SIM_STEP + max(fcst_leng) * self.DAY_STEP
        MAX_SIM_IDX = MAX_SIM_STEP + 1
        self.x_tru = np.zeros((MAX_SIM_IDX, self.N))
        self.x_tru[0, :] = np.load('./data/x_tru_init.npy')
        for t in range(MAX_SIM_STEP) :
            self.x_tru[t+1, :] = self.runge_kutta(self.x_tru[t, :])
        for j, f in enumerate(fcst_leng) :
            FCST_STEP = f * self.DAY_STEP
            SIM_IDX = self.SIM_IDX - self.WINDOW_DAY_STEP + FCST_STEP
            x_a = np.zeros((SIM_IDX, self.N))
            x_b_L = np.zeros((SIM_IDX, self.N))
            for i in range(self.WINDOW_NUM) :
                window_init_index = i * self.WINDOW_DAY_STEP
                window_next_index = window_init_index + FCST_STEP
                x_a[window_init_index, :] = x_opt[i, :]
                for t in range(window_init_index, window_next_index) :
                    x_a[t+1, :] = self.runge_kutta(x_a[t, :])
                x_b_L[window_next_index, :] = x_a[window_next_index, :]
            first_index = FCST_STEP
            final_index = SIM_IDX
            interval = self.WINDOW_DAY_STEP
            square_diff = np.square(x_b_L[first_index:final_index:interval] - self.x_tru[first_index:final_index:interval])
            rmse = np.sqrt(np.average(square_diff, axis=1))
            rmse_avg = np.average(rmse)
            rmse_list.append(rmse)
            rmse_avg_list[j] = rmse_avg
        return rmse_avg_list
        
    def make_data(self) :
        rmse_avg_list = self.calculate_rmse()
        np.save("./data/RMSE_SIM", rmse_avg_list)

    def sensitivity_scaling_factor(self) :
        scaling_factor_list = [5, 10, 20, 50, 100, 200, 500]
        rmse_scaling_factor = np.zeros((len(scaling_factor_list)))
        for i, sf in enumerate(scaling_factor_list) :
            self.quantum_annealing(SCALING_FACTOR=sf)
            rmse_avg_list = self.calculate_rmse()
            rmse_scaling_factor[i] = rmse_avg_list[0]
        np.save("./data/RMSE_SCALING_FACTOR_SIM", rmse_scaling_factor)


qa = Annealing()
qa.sensitivity_scaling_factor()
qa.quantum_annealing()
qa.make_data()
