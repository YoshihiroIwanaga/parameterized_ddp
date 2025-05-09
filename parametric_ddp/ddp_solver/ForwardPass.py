import numpy as np

from parametric_ddp.ddp_solver.BackwardPass import BackwardPassResult
from parametric_ddp.utils.OptimalControlProblem import OptimalControlProblem
from parametric_ddp.utils.TrajInfo import TrajInfo
from problem_setting.abstract.Config import Config


class ForwardPass:
    def __init__(self, ocp: OptimalControlProblem, cfg: Config) -> None:
        self.ocp: OptimalControlProblem = ocp
        self.horizon: int = cfg.horizon
        self.n: int = cfg.n
        self.m: int = cfg.m
        self.step_sizes: np.ndarray = (
            2 ** np.linspace(0, -14, cfg.step_size_num)  # type: ignore
        ).reshape((cfg.step_size_num, 1))
        self.free_state_idx: list[int] = cfg.free_state_idx
        self.failed: bool = False

    def _update_traj(
        self, traj: TrajInfo, traj_pre: TrajInfo, bp_result: BackwardPassResult
    ) -> None:
        for t in range(self.horizon):
            if t == 0:
                if len(self.free_state_idx) > 0:
                    traj.set_x0(
                        traj_pre.xs[
                            traj_pre.traj_idx : traj_pre.traj_idx + 1,
                            :,
                            t,
                        ]
                        + self.step_sizes * bp_result.kx0
                    )
                traj.us[:, :, t] = (
                    traj_pre.us[traj_pre.traj_idx : traj_pre.traj_idx + 1, :, t]
                    + self.step_sizes * bp_result.ku[:, t]
                )
            else:
                delta_x = (
                    traj.xs[:, :, t : t + 1]
                    - traj_pre.xs[
                        traj_pre.traj_idx : traj_pre.traj_idx + 1, :, t : t + 1
                    ]
                )
                traj.us[:, :, t] = (
                    traj_pre.us[traj_pre.traj_idx : traj_pre.traj_idx + 1, :, t]
                    + self.step_sizes * bp_result.ku[:, t]
                    + (bp_result.Ku[:, :, t] @ delta_x)[:, :, 0]
                )
            self.ocp.transit(traj, t)
        self.ocp.calc_cost(traj)

    def _select_best_traj(self, traj: TrajInfo) -> None:
        traj.set_traj_idx(np.nanargmin(traj.costs))  # type:ignore

    def update(
        self,
        traj: TrajInfo,
        traj_pre: TrajInfo,
        bp_result: BackwardPassResult,
    ) -> None:
        self.failed = False
        self._update_traj(traj, traj_pre, bp_result)
        if np.all(np.isinf(traj.costs)):
            self.failed = True
            return
        self._select_best_traj(traj)
