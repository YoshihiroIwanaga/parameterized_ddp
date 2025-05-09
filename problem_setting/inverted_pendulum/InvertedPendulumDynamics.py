import numpy as np

from parametric_ddp.utils.TrajInfo import TrajInfo
from problem_setting.abstract.Config import Config
from problem_setting.abstract.Dynamics import Dynamics


class InvertedPendulumDynamics(Dynamics):
    def __init__(self, config: Config) -> None:
        super(InvertedPendulumDynamics, self).__init__(config)
        self.has_nonzero_fxx: bool = True
        self.has_nonzero_fxu: bool = False
        self.has_nonzero_fuu: bool = False

    def set_constant(
        self,
        traj_info: TrajInfo,
    ) -> None:
        # fxs: np.ndarray,  # ∈R^{n,n,te}
        # fus: np.ndarray,  # ∈R^{n,m,te}
        # fxs: np.ndarray,  # ∈R^{n,n,te}
        traj_info.fxs[0, 0, :] = 1
        traj_info.fxs[0, 1, :] = self.dt
        traj_info.fxs[1, 1, :] = 1
        # fus: np.ndarray,  # ∈R^{n,m,te}
        traj_info.fus[1, 0, :] = self.dt

    def transit(
        self,
        traj_info: TrajInfo,
        t: int,  # time step
    ) -> None:
        # xs: np.ndarray,  # ∈R^{K,n,te+1} K:number of step sizes
        # us: np.ndarray,  # ∈R^{K,m,te} K:number of step sizes

        # φ[k+1] = φ[k] + dt*ω[k])
        traj_info.xs[:, 0, t + 1 : t + 2] = (
            traj_info.xs[:, 0, t : t + 1] + self.dt * traj_info.xs[:, 1, t : t + 1]
        )

        # ω[k+1] = ω[k] + dt*(sin(φ[k]) + u[k])
        traj_info.xs[:, 1, t + 1 : t + 2] = traj_info.xs[:, 1, t : t + 1] + self.dt * (
            np.sin(traj_info.xs[:, 0, t : t + 1]) + traj_info.us[:, 0, t : t + 1]
        )

    def calc_jacobian(
        self,
        traj_info: TrajInfo,
    ) -> None:
        cos_phi = np.cos(traj_info.xs[traj_info.traj_idx, 0, :-1])
        traj_info.fxs[1, 0, :] = self.dt * cos_phi

    def calc_hessian(
        self,
        traj_info: TrajInfo,
    ) -> None:
        sin_phi = np.sin(traj_info.xs[traj_info.traj_idx, 0, :-1])
        traj_info.fxxs[1, 0, 0, :] = -self.dt * sin_phi
