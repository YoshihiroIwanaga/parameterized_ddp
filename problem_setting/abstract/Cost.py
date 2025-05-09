import numpy as np

from parametric_ddp.utils.TrajInfo import TrajInfo
from problem_setting.abstract.Config import Config


class Cost:
    def __init__(self, cfg: Config) -> None:
        self.n = cfg.n
        self.m = cfg.m
        self.x_ref: np.ndarray = cfg.x_ref
        self.horizon = cfg.horizon
        self.Q_ini: np.ndarray = cfg.Q_ini
        self.Q: np.ndarray = cfg.Q
        self.R: np.ndarray = cfg.R
        self.Q_terminal: np.ndarray = cfg.Q_terminal
        self.Qs = np.zeros(
            (
                self.horizon * self.n,
                self.horizon * self.n,
            )
        )
        self.Rs = np.zeros(
            (
                self.horizon * self.m,
                self.horizon * self.m,
            )
        )
        for i in range(self.horizon):
            if i == 0:
                self.Qs[
                    self.n * i : self.n * (i + 1),
                    self.n * i : self.n * (i + 1),
                ] = self.Q_ini
            else:
                self.Qs[
                    self.n * i : self.n * (i + 1),
                    self.n * i : self.n * (i + 1),
                ] = self.Q
            self.Rs[
                self.m * i : self.m * (i + 1),
                self.m * i : self.m * (i + 1),
            ] = self.R

        self.diag_Qs = np.diag(self.Qs)
        self.diag_Rs = np.diag(self.Rs)

        self.diag_Q_terminal = np.diag(self.Q_terminal)
        self.evaluates_Q: bool = np.any(self.diag_Qs)  # type:ignore
        self.evaluates_R: bool = np.any(self.diag_Rs)  # type:ignore

    def set_constant(
        self,
        traj_info: TrajInfo,
    ) -> None:
        for i in range(self.horizon):
            if i == 0:
                traj_info.stage_cost_xxs[:, :, i] = self.Q_ini
            else:
                traj_info.stage_cost_xxs[:, :, i] = self.Q
        for i in range(self.horizon):
            traj_info.stage_cost_uus[:, :, i] = self.R
        traj_info.terminal_cost_xx[:, :] = self.Q_terminal

    def calc_stage_cost(
        self,
        traj_info: TrajInfo,
    ) -> None:
        traj_info.stage_costs[:, :] = 0.0

        if self.evaluates_Q:
            states_reshaped = (
                (
                    traj_info.xs[:, :, :-1] - self.x_ref.reshape((1, self.n, 1))
                ).T.reshape((self.n * (self.horizon), -1))
            ).T  # ∈R^{K,n*te}
            Q_cost = 1 / 2 * self.diag_Qs * states_reshaped * states_reshaped
            Q_cost = Q_cost.reshape((-1, self.horizon, self.n))
            traj_info.stage_costs[:, :] += np.sum(Q_cost, axis=2)

        if self.evaluates_R:
            inputs_reshaped = (traj_info.us.T.reshape((self.m * (self.horizon), -1))).T
            R_cost = 1 / 2 * self.diag_Rs * inputs_reshaped * inputs_reshaped
            R_cost = R_cost.reshape((-1, self.horizon, self.m))
            traj_info.stage_costs[:, :] += np.sum(R_cost, axis=2)

    def calc_terminal_cost(
        self,
        traj_info: TrajInfo,
    ) -> None:
        states_reshaped = (traj_info.xs[:, :, -1:].T.reshape((self.n, -1))).T
        tmp = (
            1
            / 2
            * self.diag_Q_terminal
            * (states_reshaped - self.x_ref)
            * (states_reshaped - self.x_ref)
        )
        traj_info.terminal_costs[:] = np.sum(tmp, axis=1)

    def calc_grad(
        self,
        traj_info: TrajInfo,
    ) -> None:
        if self.evaluates_Q:
            traj_info.stage_cost_xs[:, :, :] = (
                (
                    self.diag_Qs
                    * (
                        traj_info.xs[traj_info.traj_idx, :, :-1]
                        - self.x_ref.reshape((self.n, 1))
                    ).T.reshape(self.n * self.horizon)
                )
                .reshape(self.horizon, 1, self.n)
                .T
            )
        else:
            traj_info.stage_cost_xs[:, :, :] = 0.0

        if self.evaluates_R:
            traj_info.stage_cost_us[:, :, :] = (
                (
                    self.diag_Rs
                    * traj_info.us[traj_info.traj_idx, :, :].T.reshape(
                        self.m * self.horizon
                    )
                )
                .reshape(self.horizon, 1, self.m)
                .T
            )
        else:
            traj_info.stage_cost_us[:, :, :] = 0.0

        traj_info.terminal_cost_x[:] = self.diag_Q_terminal * (
            traj_info.xs[traj_info.traj_idx, :, -1] - self.x_ref
        )

    def calc_hessian(self, traj_info: TrajInfo) -> None:
        pass
