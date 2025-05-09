import numpy as np


class Config(object):
    def __init__(self) -> None:
        self.n: int = 0  # state dimension
        self.m: int = 0  # control input dimension
        self.dt: float = 0.01
        self.horizon: int = 100  # prediction horizon
        self.x_ini: np.ndarray = np.zeros((self.n,))  # initial state
        self.x_ref: np.ndarray = np.zeros((self.n,))  # target state
        # weight matrices of cost function
        self.Q_ini: np.ndarray = np.diag(np.zeros((self.n)))
        self.Q: np.ndarray = np.diag(np.zeros((self.n)))
        self.R: np.ndarray = np.diag(np.zeros((self.m)))
        self.P = np.zeros((self.n, self.m))
        self.Q_terminal: np.ndarray = np.diag(np.zeros((self.n)))
        # for DDP
        self.tol: float = 1e-7
        self.max_iter: int = 1000
        self.run_ddp: bool = False #iLQR or DDP
        self.step_size_num: int = 21
        self.cost_change_ratio_convergence_criteria: float = 0.999
        self.u_ini: np.ndarray = np.zeros((self.m, self.horizon))
        self.u_inis: np.ndarray = np.zeros((self.step_size_num, self.m, self.horizon))
        self.free_state_idx: list[int] = []
        self.normalize_Qu_error: bool = False
        self.Qu_normalization_factor: float = 1.0
        self.knows_optimal_cost: bool = False
        self.optimal_cost: float = 0.0
