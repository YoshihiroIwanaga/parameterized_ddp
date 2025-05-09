import typing

import numpy as np

from parametric_ddp.interpolator.cubic.CubicInterpolationConfigWrapper import (
    CubicInterpolationConfigWrapper,
)
from parametric_ddp.utils.OptimalControlProblem import OptimalControlProblem
from parametric_ddp.utils.TrajInfoWrapper import TrajInfoWrapper
from problem_setting.abstract.Cost import Cost
from problem_setting.abstract.Dynamics import Dynamics


class CubicInterpolationWrapper(OptimalControlProblem):
    def __init__(
        self,
        dynamics: Dynamics,
        cost: Cost,
        cfg: CubicInterpolationConfigWrapper,
    ) -> None:
        self.dynamics: Dynamics = dynamics
        self.cost: Cost = cost
        self.cfg: CubicInterpolationConfigWrapper = cfg
        self.n: int = cfg.n
        self.m: int = cfg.m
        self.n_org: int = cfg.org_cfg.n
        self.m_org: int = cfg.org_cfg.m
        self.horizon: int = cfg.horizon
        self.steps_per_knot: typing.List[int] = (
            cfg.steps_per_knot
        )  # [L_0, L_1,..., L_{Te-1}] ∈ N^{Te}
        self.steps_per_knot_array: np.ndarray = np.array(
            self.steps_per_knot
        )  # [L_0, L_1,..., L_{Te-1}] ∈ N^{Te}
        self.time_step_on_knot: typing.List[int] = (
            cfg.time_step_on_knot
        )  # [k_(0), k_(1),..., k_(Te)] ∈ N^{Te+1}
        self.knot_idxs: typing.List[int] = (
            cfg.knot_idxs
        )  # [0,..,0,1,...,1,...,Te,...,Te] ∈ N^{te}
        self.time_step_from_knots: typing.List[int] = (
            cfg.time_step_from_knots
        )  # [0,1..,L_0-1,0,1...,L_1-1,...,0,1,...,L_{Te}-1] ∈ N^{te}

        self.LT_dt = self.cfg.dt * self.steps_per_knot_array
        # u[k+l] = g(s[k], ν[k], l) = g_l(s[k], ν[k])
        # ∂u[k+l]/∂s[k] = ∂g_l/∂s[k]
        # self.gss_const contains constant part of matrices respresenting ∂g_l/∂s[k] for k = 0...Te.
        self.gss_const = np.zeros(
            (
                self.cfg.org_cfg.m,
                2 * self.cfg.org_cfg.m,
                self.cfg.org_cfg.horizon,
            )
        )
        # self.gvs_const contains contant part of  matrices respresenting ∂g_l/∂ν[k] for k = 0...Te.
        self.gvs_const = np.zeros(
            (
                self.cfg.org_cfg.m,
                self.cfg.org_cfg.m,
                self.cfg.org_cfg.horizon,
            )
        )
        cnt = 0
        for k in range(self.cfg.horizon):
            for i in range(self.cfg.steps_per_knot[k]):
                # u[k+l] = g(s[k], ν[k], l) = g_l(s[k], ν[k])
                # ∂u[k+l]/∂s[k] = ∂g_l/∂s[k]
                self.gss_const[:, :, cnt] = np.hstack(
                    (
                        np.eye(self.cfg.org_cfg.m),
                        i * self.cfg.dt * np.eye(self.cfg.org_cfg.m),
                    )
                )
                # ∂u[k+l]/∂ν[k] = ∂g_l/∂ν[k]
                self.gvs_const[:, :, cnt] = (
                    (self.cfg.dt**2) * (i**2 / 2) * np.eye(self.cfg.org_cfg.m)
                )
                cnt += 1
        self.gss_nonconst = np.zeros(
            (
                self.cfg.org_cfg.m,
                2 * self.cfg.org_cfg.m,
            )
        )
        self.gvs_nonconst = np.zeros(
            (
                self.cfg.org_cfg.m,
                self.cfg.org_cfg.m,
            )
        )
        # u[k+l] = p[k] + a0(l)*δ[k] + a1(l)*μ[k] + a2(l)*δ[k]*η[k] + a3(l)*μ[k]*η[k]
        self.a0s = np.zeros((self.horizon, max(self.steps_per_knot)))
        self.a1s = np.zeros((self.horizon, max(self.steps_per_knot)))
        self.a2s = np.zeros((self.horizon, max(self.steps_per_knot)))
        self.a3s = np.zeros((self.horizon, max(self.steps_per_knot)))
        for k in range(self.horizon):
            L_T = self.steps_per_knot[k]
            for i in range(L_T):
                self.a0s[k, i] = i * self.cfg.dt
                self.a1s[k, i] = 0.5 * (self.cfg.dt**2) * (i**2)
                self.a2s[k, i] = (3 / L_T * i**2 - 2 / (L_T**2) * i**3) * self.cfg.dt
                self.a3s[k, i] = self.cfg.dt**2 * (3 / 2 * (i**2) - i**3 / L_T)

    def set_constant(
        self,
        traj: TrajInfoWrapper,
    ) -> None:
        self.dynamics.set_constant(traj.every_traj)
        self.cost.set_constant(traj)
        # inter-knot dynamics is given by
        #                  | f(f(f...f(x, u)))           | } n
        # F_T(X_T, U_T) =  | p + L_T*Δt*(δ+L_T*Δt*μ/2)  | } m
        #                  | δ + L_T*Δt*μ               | } m
        # therfore,
        #       | *   *     *    | } n
        # F_X = | 0   I  L_T*Δt  | } original m
        #       | 0   0     I    | } original m
        for i in range(2 * self.m):
            traj.fxs[self.cfg.org_cfg.n + i, self.cfg.org_cfg.n + i, :] = 1.0
        for i, step_per_knot in enumerate(self.steps_per_knot):
            traj.fxs[
                self.cfg.org_cfg.n : self.cfg.org_cfg.n + self.cfg.org_cfg.m,
                self.cfg.org_cfg.n + self.cfg.org_cfg.m :,
                i,
            ] = (
                self.cfg.dt * step_per_knot * np.eye(self.cfg.org_cfg.m)
            )
        #       |          *         | } n
        # F_U = |  0.5*L_T^2*Δt^2    | } original m
        #       |      L_T*Δt        | } original m
        for i, step_per_knot in enumerate(self.steps_per_knot):
            traj.fus[-self.m :, :, i] = self.cfg.dt * step_per_knot * np.eye(self.m)
            traj.fus[
                self.cfg.org_cfg.n : self.cfg.org_cfg.n + self.cfg.org_cfg.m,
                :,
                i,
            ] = (
                0.5 * (self.cfg.dt * step_per_knot) ** 2 * np.eye(self.cfg.org_cfg.m)
            )
        traj.stage_cost_xxs = self.steps_per_knot_array * traj.stage_cost_xxs
        traj.stage_cost_uus = self.steps_per_knot_array * traj.stage_cost_uus
        traj.stage_cost_xus = self.steps_per_knot_array * traj.stage_cost_xus

    def transit(
        self,
        traj: TrajInfoWrapper,
        k: int,  # knot step
    ) -> None:
        p_ = traj.xs[:, self.cfg.org_cfg.n : self.cfg.org_cfg.n + self.cfg.org_cfg.m, k]
        delta = traj.xs[:, self.cfg.org_cfg.n + self.cfg.org_cfg.m :, k]
        mu = traj.us[:, :, k]
        LT_dt = self.LT_dt[k]

        # p[k+1] = p[k] + Δt・L[k]・(δ[k] + 0.5*L_T*Δt*μ[k])/2  ---(1)
        traj.xs[
            :,
            self.cfg.org_cfg.n : self.cfg.org_cfg.n + self.cfg.org_cfg.m,
            k + 1,
        ] = p_ + LT_dt * (delta + 0.5 * LT_dt * mu)

        # δ[k+1] = δ[k] +  Δt・L[k]・μ[k]  --------------------------(2)
        traj.xs[:, self.cfg.org_cfg.n + self.cfg.org_cfg.m :, k + 1] = (
            delta + LT_dt * mu
        )

        for i in range(self.steps_per_knot[k]):
            # H1(t) = 2*t**3 - 3*t**2 + 1
            # H2(t) = t**3 - 2*t**2 + t
            # H3(t) = -2*t**3 + 3*t**2
            # H4(t) = t**3 - t**2
            # t　∈ [0, 1]
            # u = H1(t)*p[k] + H2(t)*LΔt*δ[k] +  H3(t)*p[k+1] + H4(t)*LΔt*δ[k+1]
            # By substituting (1), (2) and t = i/L
            # u can be written as following
            traj.every_traj.us[:, :, self.time_step_on_knot[k] + i] = (
                p_ + self.a0s[k, i] * delta + self.a1s[k, i] * mu
            )
            self.dynamics.transit(
                traj.every_traj,
                self.time_step_on_knot[k] + i,
            )
        # X[T+1] = x(k(T) + L_T)
        traj.xs[:, : self.cfg.org_cfg.n, k + 1] = traj.every_traj.xs[
            :,
            : self.cfg.org_cfg.n,
            self.time_step_on_knot[k] + self.steps_per_knot[k],
        ]

    def calc_cost(self, traj: TrajInfoWrapper) -> None:
        # eval cost at knot step
        self.cost.calc_stage_cost(traj)
        self.cost.calc_terminal_cost(traj)
        traj.costs = (
            np.sum(self.steps_per_knot_array * traj.stage_costs, axis=1)
            + traj.terminal_costs
        )

    def _reset_diff(self, traj: TrajInfoWrapper, t: int, calc_dynamics_hessian: bool):
        # ∂x[k+1]/∂x[k] ∈ R^{n*n}
        ax_ax = traj.every_traj.fxs[:, :, t]
        # ∂x[k+1]/∂s[k] = ∂x[k+1]/∂u[k] ∂u[k]/∂s[k] ∈ R^{n*m}
        ax_as = traj.every_traj.fus[:, :, t] @ self.gss_const[:, :, t]
        # ∂x[k+1]/∂ν[k] ∈ R^{n*m}
        ax_av = np.zeros(
            (
                self.n_org,
                self.m_org,
            )
        )
        if calc_dynamics_hessian:
            a2x_axx_reshaped = traj.every_traj.fxxs[:, :, :, t].reshape(self.n_org, -1)
            if self.dynamics.has_nonzero_fxu:
                a2x_axs_reshaped = np.einsum(
                    "gj,kig->kij",
                    self.gss_const[:, :, t],
                    traj.every_traj.fxus[:, :, :, t],
                ).reshape(self.n_org, -1)
            else:
                a2x_axs_reshaped = np.zeros((self.n_org, self.n_org * 2 * self.m_org))
            if self.dynamics.has_nonzero_fuu:
                a2x_ass_reshaped = self._einsum_separate(
                    self.gss_const[:, :, t],
                    traj.every_traj.fuus[:, :, :, t],
                    self.gss_const[:, :, t],
                ).reshape(self.n_org, -1)
            else:
                a2x_ass_reshaped = np.zeros(
                    (
                        self.cfg.org_cfg.n,
                        4 * self.cfg.org_cfg.m * self.cfg.org_cfg.m,
                    )
                )

            a2x_avv_reshaped = np.zeros((self.n_org, self.m_org * self.m_org))
            a2x_axv_reshaped = np.zeros((self.n_org, self.n_org * self.m_org))
            a2x_asv_reshaped = np.zeros((self.n_org, 2 * self.m_org * self.m_org))
        else:
            a2x_axx_reshaped = np.zeros(0)
            a2x_axs_reshaped = np.zeros(0)
            a2x_ass_reshaped = np.zeros(0)
            a2x_avv_reshaped = np.zeros(0)
            a2x_axv_reshaped = np.zeros(0)
            a2x_asv_reshaped = np.zeros(0)
        return (
            ax_ax,
            ax_as,
            ax_av,
            a2x_axx_reshaped,
            a2x_axs_reshaped,
            a2x_ass_reshaped,
            a2x_avv_reshaped,
            a2x_axv_reshaped,
            a2x_asv_reshaped,
        )


    def diff(self, traj: TrajInfoWrapper, calc_dynamics_hessian: bool) -> None:
        # calc fx, fu at every time steps
        self.dynamics.calc_jacobian(traj.every_traj)
        if calc_dynamics_hessian:
            self.dynamics.calc_hessian(traj.every_traj)
        # calc cx, cu, qx, qu, qxx, quu at knot steps
        self.cost.calc_grad(traj)
        self.cost.calc_hessian(traj)
        traj.stage_cost_xs = self.steps_per_knot_array * traj.stage_cost_xs
        traj.stage_cost_us = self.steps_per_knot_array * traj.stage_cost_us

        for k in range(self.cfg.horizon):

            for i in range(self.cfg.steps_per_knot[k]):
                t = self.time_step_on_knot[k] + i
                # u[k+l] = p[k] + a0(l)*δ[k] + a1(l)*μ[k] + a2(l)*δ[k]*η[k] + a3(l)*μ[k]*η[k]
                # ∂u[k+l]/∂s[k] = |I, a0(l)| + |0, a2(l)*diag(η[k])|
                gss = self.gss_const[:, :, t]
                # ∂u[k+l]/∂ν[k] = |a1(l)| + |a3(l)*diag(η[k])|
                gvs = self.gvs_const[:, :, t]

                if i == 0:
                    (
                        ax_ax,
                        ax_as,
                        ax_av,
                        a2x_axx_reshaped,
                        a2x_axs_reshaped,
                        a2x_ass_reshaped,
                        a2x_avv_reshaped,
                        a2x_axv_reshaped,
                        a2x_asv_reshaped,
                    ) = self._reset_diff(traj, t, calc_dynamics_hessian)
                else:
                    # ∂x[k+i+1]/∂x[k] = ∂f/∂x[k+i] @ ∂x[k+i]/∂x[k]
                    ax_ax_ = traj.every_traj.fxs[:, :, t] @ ax_ax  # type:ignore
                    # ∂x[k+i+1]/∂s[k] = ∂f/∂x[k+i] @ ∂x[k+i]/∂s[k] + ∂f/∂u[k+i] @ ∂u[k+i]/∂s[k]
                    ax_as_ = (
                        traj.every_traj.fxs[:, :, t] @ ax_as  # type:ignore
                        + traj.every_traj.fus[:, :, t] @ gss
                    )
                    # ∂x[k+i+1]/∂ν[k] = ∂f/∂x[k+i] @ ∂x[k+i]/∂ν[k] + ∂f/∂u[k+i] @ ∂u[k+i]/∂ν[k]
                    ax_av_ = (
                        traj.every_traj.fxs[:, :, t] @ ax_av  # type:ignore
                        + traj.every_traj.fus[:, :, t] @ gvs
                    )
                    if calc_dynamics_hessian:
                        # a2x_axx_reshaped = a2x_axx.reshape(self.cfg.org_cfg.n, -1)
                        a2x_axx_reshaped = self._calc_a2x_axx(
                            a2x_axx_reshaped,
                            ax_ax,
                            traj.every_traj.fxs[:, :, t],
                            traj.every_traj.fxxs[:, :, :, t],
                        )
                        a2x_axs_reshaped = self._calc_a2x_axs(
                            a2x_axs_reshaped,
                            ax_ax,
                            ax_as,
                            gss,
                            traj.every_traj.fxs[:, :, t],
                            traj.every_traj.fxxs[:, :, :, t],
                            traj.every_traj.fxus[:, :, :, t],
                        )
                        a2x_ass_reshaped = self._calc_a2x_ass(
                            a2x_ass_reshaped,
                            ax_as,
                            gss,
                            traj.every_traj.fxs[:, :, t],
                            traj.every_traj.fxxs[:, :, :, t],
                            traj.every_traj.fxus[:, :, :, t],
                            traj.every_traj.fuus[:, :, :, t],
                        )
                        a2x_avv_reshaped = self._calc_a2x_avv(
                            a2x_avv_reshaped,
                            ax_av,
                            gvs,
                            traj.every_traj.fxs[:, :, t],
                            traj.every_traj.fus[:, :, t],
                            traj.every_traj.fxxs[:, :, :, t],
                            traj.every_traj.fxus[:, :, :, t],
                            traj.every_traj.fuus[:, :, :, t],
                        )
                        a2x_axv_reshaped = self._calc_a2x_axv(
                            a2x_axv_reshaped,
                            ax_ax,
                            ax_av,
                            gvs,
                            traj.every_traj.fxs[:, :, t],
                            traj.every_traj.fxxs[:, :, :, t],
                            traj.every_traj.fxus[:, :, :, t],
                        )
                        a2x_asv_reshaped = self._calc_a2x_asv(
                            a2x_asv_reshaped,
                            ax_av,
                            ax_as,
                            traj.every_traj.fxs[:, :, t],
                            traj.every_traj.fxxs[:, :, :, t],
                            traj.every_traj.fxus[:, :, :, t],
                        )
                    ax_ax = ax_ax_
                    ax_as = ax_as_
                    ax_av = ax_av_

            #       | ax_ax   ax_ap      ax_aδ   | } n
            # F_X = |    0      I        L*Δt*I  |  } original m
            #       |    0      0          I     |  } original m
            traj.fxs[: self.cfg.org_cfg.n, : self.cfg.org_cfg.n, k] = (
                ax_ax
            )  # type:ignore
            traj.fxs[: self.cfg.org_cfg.n, self.cfg.org_cfg.n :, k] = (
                ax_as
            )  # type:ignore

            #       |      ax_aμ       | } n
            # F_U = |  0.5*L_T^2*Δt^2  | } original m
            #       |    L_T*Δt*I      | } original m
            traj.fus[: self.cfg.org_cfg.n, :, k] = ax_av  # type:ignore

            if calc_dynamics_hessian:
                traj.fxxs[
                    : self.cfg.org_cfg.n, : self.cfg.org_cfg.n, : self.cfg.org_cfg.n, k
                ] = a2x_axx_reshaped.reshape(self.n_org, self.n_org, self.n_org)
                traj.fxxs[
                    : self.cfg.org_cfg.n, self.cfg.org_cfg.n :, self.cfg.org_cfg.n :, k
                ] = a2x_ass_reshaped.reshape(self.n_org, 2 * self.m_org, 2 * self.m_org)
                traj.fxxs[
                    : self.cfg.org_cfg.n, : self.cfg.org_cfg.n, self.cfg.org_cfg.n :, k
                ] = a2x_axs_reshaped.reshape(self.n_org, self.n_org, 2 * self.m_org)
                traj.fxxs[
                    : self.cfg.org_cfg.n, self.cfg.org_cfg.n :, : self.cfg.org_cfg.n, k
                ] = a2x_axs_reshaped.reshape(self.n_org, 2 * self.m_org, self.n_org)

                traj.fxus[: self.cfg.org_cfg.n, : self.cfg.org_cfg.n, :, k] = (
                    a2x_axv_reshaped.reshape(self.n_org, self.n_org, self.m_org)
                )
                traj.fxus[: self.cfg.org_cfg.n, self.cfg.org_cfg.n :, :, k] = (
                    a2x_asv_reshaped.reshape(self.n_org, 2 * self.m_org, self.m_org)
                )
                traj.fuus[: self.cfg.org_cfg.n, :, :, k] = a2x_avv_reshaped.reshape(
                    self.n_org, self.m_org, self.m_org
                )

    def _calc_a2x_axx(
        self,
        a2x_axx_reshaped: np.ndarray,
        ax_ax: np.ndarray,
        fx: np.ndarray,
        fxx: np.ndarray,
    ) -> np.ndarray:
        # ∂^2x[k+i+1]/∂x[k]∂x[k]
        # = ∑ ∂x[k+i]/∂x[k]*∂^2x[k+i+1]/∂x[k+i]∂x[k+i]*∂x[k+i]/∂x[k] + ∑∂x[k+i+1]/∂x[k+i]*∂^2x[k+i]/∂x[k+i-1]∂x[k+i-1]
        # a2x_axx = self._einsum_separate(
        #     ax_ax,
        #     traj.every_traj.fxxs[:, :, :, t],
        #     ax_ax,
        # ) + np.einsum(
        #     "kp,pij->kij", traj.every_traj.fxs[:, :, t], a2x_axx
        # )
        if self.dynamics.has_nonzero_fxx:
            return self._einsum_separate(
                ax_ax,
                fxx,
                ax_ax,
            ).reshape(
                self.n_org, -1
            ) + (fx @ a2x_axx_reshaped)
        else:
            return a2x_axx_reshaped

    def _calc_a2x_axs(
        self,
        a2x_axs_reshaped: np.ndarray,
        ax_ax: np.ndarray,
        ax_as: np.ndarray,
        gss: np.ndarray,
        fx: np.ndarray,
        fxx: np.ndarray,
        fxu: np.ndarray,
    ) -> np.ndarray:
        if self.dynamics.has_nonzero_fxx and self.dynamics.has_nonzero_fxu:
            return (
                self._einsum_separate(
                    ax_as,
                    fxx,
                    ax_ax,
                ).reshape(self.n_org, -1)
                + self._einsum_separate(
                    gss,  # au_as
                    fxu,
                    ax_ax,
                ).reshape(self.n_org, -1)
                + fx @ a2x_axs_reshaped
            )
        elif self.dynamics.has_nonzero_fxx:
            return (
                self._einsum_separate(
                    ax_as,
                    fxx,
                    ax_ax,
                ).reshape(self.n_org, -1)
                + fx @ a2x_axs_reshaped
            )
        else:
            return (
                self._einsum_separate(
                    gss,  # au_as
                    fxu,
                    ax_ax,
                ).reshape(self.n_org, -1)
                + fx @ a2x_axs_reshaped
            )

    def _calc_a2x_ass(
        self,
        a2x_ass_reshaped: np.ndarray,
        ax_as: np.ndarray,
        gss: np.ndarray,
        fx: np.ndarray,
        fxx: np.ndarray,
        fxu: np.ndarray,
        fuu: np.ndarray,
    ) -> np.ndarray:
        return (
            self._einsum_separate(
                ax_as,
                fxx,
                ax_as,
            ).reshape(self.n_org, -1)
            # + self._einsum_separate(
            #     gss,
            #     traj.every_traj.fxus[:, :, :, t],
            #     ax_as,
            # )  # au_as
            + fx @ a2x_ass_reshaped
            # + self._einsum_separate(
            #     gss,
            #     traj.every_traj.fxus[:, :, :, t],
            #     ax_as,
            # )
            # + self._einsum_separate(
            #     gss,
            #     traj.every_traj.fuus[:, :, :, t],
            #     gss,
            # )  # au_as
            # + np.einsum(
            #     "gw,wij->gij",
            #     traj.every_traj.fus[:, :, t],
            #     self.gss[:,:,:,t],
            # )  # a2u_as2
        )

    def _calc_a2x_avv(
        self,
        a2x_avv_reshaped: np.ndarray,
        ax_av: np.ndarray,
        gvs: np.ndarray,
        fx: np.ndarray,
        fu: np.ndarray,
        fxx: np.ndarray,
        fxu: np.ndarray,
        fuu: np.ndarray,
    ) -> np.ndarray:
        return (
            self._einsum_separate(
                ax_av,
                fxx,
                ax_av,
            ).reshape(self.n_org, -1)
            # + self._einsum_separate(
            #     gvs,
            #     traj.every_traj.fxus[:, :, :, t],
            #     ax_av,
            # )  # au_av
            + fx @ a2x_avv_reshaped
            # + self._einsum_separate(
            #     gvs,
            #     traj.every_traj.fxus[:, :, :, t],
            #     ax_av,
            # )
            # + self._einsum_separate(
            #     gvs,
            #     traj.every_traj.fuus[:, :, :, t],
            #     gvs,
            # )  # au_as
            # + np.einsum(
            #     "gw,wij->gij",
            #     traj.every_traj.fus[:, :, t],
            #     self.gvv[:, :, :, t],
            # )  # a2u_av2
        )

    def _calc_a2x_axv(
        self,
        a2x_axv_reshaped: np.ndarray,
        ax_ax: np.ndarray,
        ax_av: np.ndarray,
        gvs,
        fx: np.ndarray,
        fxx: np.ndarray,
        fxu: np.ndarray,
    ) -> np.ndarray:
        return (
            self._einsum_separate(
                ax_av,
                fxx,
                ax_ax,
            ).reshape(self.n_org, -1)
            # + self._einsum_separate(
            #     gvs,
            #     traj.every_traj.fxus[:, :, :, t],
            #     ax_ax,
            # )  # au_av
            + fx @ a2x_axv_reshaped
        )

    def _calc_a2x_asv(
        self,
        a2x_asv_reshaped: np.ndarray,
        ax_av: np.ndarray,
        ax_as,
        fx: np.ndarray,
        fxx: np.ndarray,
        fxu: np.ndarray,
    ):
        return (
            self._einsum_separate(
                ax_av,
                fxx,
                ax_as,
            ).reshape(self.n_org, -1)
            # + self._einsum_separate(
            #     gvs,
            #     traj.every_traj.fxus[:, :, :, t],
            #     ax_as,
            # )  # au_as
            + fx @ a2x_asv_reshaped
            # + self._einsum_separate(
            #     gss,
            #     traj.every_traj.fxus[:, :, :, t],
            #     ax_av,
            # )
            # + self._einsum_separate(
            #     gvs,
            #     traj.every_traj.fuus[:, :, :, t],
            #     gss,
            # )  # au_as
            # ここ0でいい？
            # + np.einsum(
            #     "gw,wij->gij",
            #     traj.every_traj.fus[:, :, t],
            #     gvs,
            # )  # a2u_asau
        )
