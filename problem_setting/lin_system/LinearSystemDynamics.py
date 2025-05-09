import numpy as np

from parametric_ddp.utils.TrajInfo import TrajInfo
from problem_setting.abstract.Config import Config
from problem_setting.abstract.Dynamics import Dynamics


class LinearSystemDynamics(Dynamics):
    def __init__(self, config: Config) -> None:
        super(LinearSystemDynamics, self).__init__(config)
        self.A = np.eye(config.n)
        self.A[0, 1] = config.dt
        self.A[2, 3] = config.dt
        self.A += 0.0 * np.random.randn(config.n, config.n)
        self.B = np.array([[0, 0], [config.dt, 0], [0, 0], [0, config.dt]])
        self.horizon = config.horizon

        self.m1 = 1.0
        self.m2 = 1.5
        self.k1 = 0.2
        self.k2 = 0.1
        self.c1 = 0.05
        self.c2 = 0.075

        self.A = np.eye(4)
        self.A[0, 1] = config.dt
        self.A[1, 0] = config.dt * (-self.k1 - self.k2) / self.m1
        self.A[1, 1] = 1.0 + config.dt * (-self.c1 - self.c2) / self.m1
        self.A[1, 2] = config.dt * (self.k2) / self.m1
        self.A[1, 3] = config.dt * (self.c2) / self.m1
        self.A[2, 3] = config.dt
        self.A[3, 0] = config.dt * (self.k2) / self.m2
        self.A[3, 1] = config.dt * (self.c2) / self.m2
        self.A[3, 2] = config.dt * (-self.k2) / self.m2
        self.A[3, 3] = 1.0 + config.dt * -(self.c2) / self.m2

        self.B[1, 0] = config.dt / self.m1
        self.B[3, 1] = config.dt / self.m2

    def set_constant(
        self,
        traj_info: TrajInfo,
    ) -> None:
        # fxs: np.ndarray,  # ∈R^{n,n,te}
        # fus: np.ndarray,  # ∈R^{n,m,te}
        for i in range(self.horizon):
            traj_info.fxs[:, :, i] = self.A
            traj_info.fus[:, :, i] = self.B

    def transit(
        self,
        traj_info: TrajInfo,
        t: int,  # time step
    ) -> None:
        # xs: np.ndarray,  # ∈R^{K,n,te+1} K:number of step sizes
        # us: np.ndarray,  # ∈R^{K,m,te} K:number of step sizes
        traj_info.xs[:, :, t + 1 : t + 2] = (
            self.A @ traj_info.xs[:, :, t : t + 1]
            + self.B @ traj_info.us[:, :, t : t + 1]
        )

