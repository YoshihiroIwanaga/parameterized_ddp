import numpy as np

from problem_setting.abstract.Config import Config


class LinearSystemConfig(Config):
    def __init__(
        self,
        x_ini: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0]),
        x_ref: np.ndarray = np.array([-3.5, 0.0, 2.0, 0.0]),
        horizon: int = 500,
        dt: float = 0.01,
    ) -> None:
        self.n: int = 4
        self.m: int = 2
        self.x_ini: np.ndarray = x_ini
        self.x_ref: np.ndarray = x_ref
        self.dt: float = dt
        self.horizon: int = horizon
        self.Q: np.ndarray = np.diag(np.array([0.001, 0.001, 0.001, 0.001]))
        self.R: np.ndarray = np.diag(np.array([0.001, 0.001]))
        self.Q_terminal: np.ndarray = np.diag(
            np.array(
                [
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ]
            )
        )
        self.Q_ini = self.Q
        self.free_state_idx = []
        self.tol = 1.0e-6
        self.max_iter: int = 1000
        self.run_ddp: bool = False
        self.step_size_num: int = 21
        self.cost_change_ratio_convergence_criteria = 0.999
        self.u_inis = np.zeros((self.step_size_num, self.m, self.horizon))
        self.normalize_Qu_error = False
        self.knows_optimal_cost = False
        self.optimal_cost = 0.0
