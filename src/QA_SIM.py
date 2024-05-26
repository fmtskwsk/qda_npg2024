from amplify import BinaryPoly
from amplify import BinaryPolyArray
from amplify import BinaryMatrix
from amplify import SymbolGenerator
from amplify import BinarySymbolGenerator
from amplify import sum_poly, pair_sum, product
from amplify import decode_solution
from amplify import Solver
from amplify import BinaryQuadraticModel
from amplify.client import FixstarsClient
import numpy as np
import argparse
from tqdm import tqdm


def get_args() : 
    parser = argparse.ArgumentParser()
    parser.add_argument("bit_num", type=int)
    parser.add_argument("scaling_factor", type=float)
    parser.add_argument("timeout", type=int)
    args = parser.parse_args()
    return args

def make_data(BIT_NUM, SCALING_FACTOR, TIMEOUT) :
    VAL_NUM = 40
    BIT_VAL_NUM = BIT_NUM * VAL_NUM
    WINDOW_NUM = 50
    gen = SymbolGenerator(BinaryPoly)
    q = gen.array(BIT_VAL_NUM)
    dx_opt = np.zeros((WINDOW_NUM, VAL_NUM))
    execution_time = 0
    B_MHRHM = np.load('./data/B_MHRHM.npy')
    _2dRHM = np.load('./data/_2dRHM.npy')[:, np.newaxis, :]
    g_T = np.zeros(BIT_NUM)
    for i, j in enumerate(range(BIT_NUM-1, -1, -1)) :
        if i == 0 :
            g_T[i] = -2 ** j
        else :
            g_T[i] = 2 ** j
    G = np.zeros((VAL_NUM, BIT_VAL_NUM))
    for i in range(VAL_NUM) :
        G[i, BIT_NUM*i:BIT_NUM*(i+1)] = g_T
    for w in tqdm(range(WINDOW_NUM)) :
        b = [q[i] for i in range(BIT_VAL_NUM)]
        b_vec = np.array(b)[:, np.newaxis]
        A = (1. / SCALING_FACTOR ** 2) * G.T @ B_MHRHM[w, :, :] @ G
        u_T = (1. / SCALING_FACTOR) * _2dRHM[w, :] @ G
        f = (b_vec.T @ A @ b_vec + u_T @ b_vec)[0, 0]
        model = BinaryQuadraticModel(f)
        client = FixstarsClient()
        client.token = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        client.parameters.timeout = TIMEOUT
        solver = Solver(client)
        solusion = solver.solve(model)
        q_opt = np.array(q.decode(solusion[0].values))
        dx_opt[w, :] = (1. / SCALING_FACTOR) * G @ q_opt
        execution_time += solver.execution_time
    np.save("./data/TIME_SIM", (execution_time / 1000.) / WINDOW_NUM)
    np.save("./data/dx_opt_SIM", dx_opt)


args = get_args()
make_data(BIT_NUM=args.bit_num, SCALING_FACTOR=args.scaling_factor, TIMEOUT=args.timeout)