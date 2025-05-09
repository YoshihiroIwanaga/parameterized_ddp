import numpy as np

from problem_setting.abstract.Config import Config


class TrajInfo:
    def __init__(
        self,
        cfg: Config,
        K: int,  # number of trajectories
    ) -> None:
        self.traj_idx: int = 0
        self.xs: np.ndarray = np.zeros((K, cfg.n, cfg.horizon + 1))
        self.xs[:, :, 0] = cfg.x_ini
        self.us: np.ndarray = np.zeros((K, cfg.m, cfg.horizon))
        self.us[:, :, :] = cfg.u_inis
        self.stage_costs: np.ndarray = np.zeros((K, cfg.horizon))
        self.terminal_costs: np.ndarray = np.zeros((K,))
        self.costs: np.ndarray = np.zeros((K,))
        self.fxs: np.ndarray = np.zeros((cfg.n, cfg.n, cfg.horizon))
        self.fus: np.ndarray = np.zeros((cfg.n, cfg.m, cfg.horizon))
        self.fxxs: np.ndarray = np.zeros((cfg.n, cfg.n, cfg.n, cfg.horizon))
        self.fxus: np.ndarray = np.zeros((cfg.n, cfg.n, cfg.m, cfg.horizon))
        self.fuus: np.ndarray = np.zeros((cfg.n, cfg.m, cfg.m, cfg.horizon))
        self.stage_cost_xs: np.ndarray = np.zeros((cfg.n, 1, cfg.horizon))
        self.stage_cost_us: np.ndarray = np.zeros((cfg.m, 1, cfg.horizon))
        self.stage_cost_xxs: np.ndarray = np.zeros((cfg.n, cfg.n, cfg.horizon))
        self.stage_cost_uus: np.ndarray = np.zeros((cfg.m, cfg.m, cfg.horizon))
        self.stage_cost_xus: np.ndarray = np.zeros((cfg.n, cfg.m, cfg.horizon))
        self.terminal_cost_x: np.ndarray = np.zeros((cfg.n,))
        self.terminal_cost_xx: np.ndarray = np.zeros((cfg.n, cfg.n))
        self.every_traj = None

    def set_traj_idx(self, idx: int) -> None:
        self.traj_idx = idx

    def set_x0(self, x0: np.ndarray) -> None:
        self.xs[:, :, 0] = x0
