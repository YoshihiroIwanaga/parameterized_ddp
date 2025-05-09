import math

import numpy as np

from parametric_ddp.utils.TrajInfo import TrajInfo
from problem_setting.abstract.Config import Config
from problem_setting.abstract.Dynamics import Dynamics


class CarDynamics(Dynamics):
    def __init__(self, config: Config) -> None:
        super(CarDynamics, self).__init__(config)
        self.wheel_base: float = 2.7
        self.v_max = 5.0
        self.v_min = -5.0
        self.steer_max = math.pi / 4
        self.steer_min = -math.pi / 4
        self.margin_rate: float = 0.05
        self.has_nonzero_fxx: bool = True
        self.has_nonzero_fxu: bool = True
        self.has_nonzero_fuu: bool = True

    def _calc_barrier(self, const, epsilon) -> np.ndarray:
        val = -1 / const - 3 / epsilon - 3 * const / epsilon**2 - const**2 / epsilon**3
        val[np.where(const <= -epsilon)] = 0.0
        val[np.where(const >= 0)] = 1e11  # np.inf
        return val

    def _diff_barrier(self, const, epsilon) -> np.ndarray:
        val = np.zeros(const.shape)
        idx = np.where((-epsilon < const) & (const < 0))
        val[idx] = 1 / const[idx] ** 2 - 3 / epsilon**2 - 2 * const[idx] / epsilon**3
        val[np.where(const >= 0)] = 1e11  # np.inf
        return val
    
    def _diff_twice_barrier(self, const, epsilon) -> np.ndarray:
        val = np.zeros(const.shape)
        idx = np.where((-epsilon < const) & (const < 0))
        val[idx] = -2 / const[idx] ** 3 - 2 / epsilon**3
        val[np.where(const >= 0)] = 1e11  # np.inf
        return val

    def set_constant(
        self,
        traj_info: TrajInfo,
    ) -> None:
        # fxs: np.ndarray,  # ∈R^{n,n,te}
        traj_info.fxs[0, 0, :] = 1
        traj_info.fxs[1, 1, :] = 1
        traj_info.fxs[2, 2, :] = 1
        traj_info.fxs[3, 3, :] = 1
        traj_info.fxs[4, 4, :] = 1
        # fus: np.ndarray,  # ∈R^{n,m,te}
        traj_info.fus[3, 0, :] = self.dt
        traj_info.fus[4, 1, :] = self.dt

    def transit(
        self,
        traj_info: TrajInfo,
        t: int,  # time step
    ) -> None:
        # xs: np.ndarray,  # ∈R^{K,n,te+1} K:number of step sizes
        # us: np.ndarray,  # ∈R^{K,m,te} K:number of step sizes

        # x[k+1] = x[k] + dt*v[k]*cos(φ[k])
        traj_info.xs[:, 0, t + 1 : t + 2] = traj_info.xs[
            :, 0, t : t + 1
        ] + self.dt * traj_info.xs[:, 3, t : t + 1] * np.cos(
            traj_info.xs[:, 2, t : t + 1]
        )
        # y[k+1] = y[k] + dt*v[k]*sin(φ[k])
        traj_info.xs[:, 1, t + 1 : t + 2] = traj_info.xs[
            :, 1, t : t + 1
        ] + self.dt * traj_info.xs[:, 3, t : t + 1] * np.sin(
            traj_info.xs[:, 2, t : t + 1]
        )
        # φ[k+1] = φ[k] + dt*v[k]*tan(δ[k])/L
        traj_info.xs[:, 2, t + 1 : t + 2] = (
            traj_info.xs[:, 2, t : t + 1]
            + self.dt
            * traj_info.xs[:, 3, t : t + 1]
            * np.tan(traj_info.xs[:, 4, t : t + 1])
            / self.wheel_base
        )
        # v[k+1] = v[k] + dt*u0[k]
        traj_info.xs[:, 3, t + 1 : t + 2] = (
            traj_info.xs[:, 3, t : t + 1] + self.dt * traj_info.us[:, 0, t : t + 1]
        )
        # δ[k+1] = δ[k] + dt*u1[k]
        traj_info.xs[:, 4, t + 1 : t + 2] = (
            traj_info.xs[:, 4, t : t + 1] + self.dt * traj_info.us[:, 1, t : t + 1]
        )
        # DBaS for box constraint
        barrier_state = 0.0
        barrier_state += self._calc_barrier(
            traj_info.xs[:, 3, t + 1 : t + 2] - self.v_max,
            self.margin_rate * self.v_max,
        )
        barrier_state += self._calc_barrier(
            -traj_info.xs[:, 3, t + 1 : t + 2] + self.v_min,
            -self.margin_rate * self.v_min,
        )
        barrier_state += self._calc_barrier(
            traj_info.xs[:, 4, t + 1 : t + 2] - self.steer_max,
            self.margin_rate * self.steer_max,
        )
        barrier_state += self._calc_barrier(
            -traj_info.xs[:, 4, t + 1 : t + 2] + self.steer_min,
            -self.margin_rate * self.steer_min,
        )
        traj_info.xs[:, 5, t + 1 : t + 2] = barrier_state

    def calc_jacobian(
        self,
        traj_info: TrajInfo,
    ) -> None:
        sin_yaw = np.sin(traj_info.xs[traj_info.traj_idx, 2, :-1])
        cos_yaw = np.cos(traj_info.xs[traj_info.traj_idx, 2, :-1])
        traj_info.fxs[0, 2, :] = (
            -self.dt * traj_info.xs[traj_info.traj_idx, 3, :-1] * sin_yaw
        )
        traj_info.fxs[0, 3, :] = self.dt * cos_yaw
        traj_info.fxs[1, 2, :] = (
            self.dt * traj_info.xs[traj_info.traj_idx, 3, :-1] * cos_yaw
        )
        traj_info.fxs[1, 3, :] = self.dt * sin_yaw
        traj_info.fxs[2, 3, :] = (
            self.dt * np.tan(traj_info.xs[traj_info.traj_idx, 4, :-1]) / self.wheel_base
        )
        traj_info.fxs[2, 4, :] = (
            self.dt
            * (
                traj_info.xs[traj_info.traj_idx, 3, :-1]
                / np.cos(traj_info.xs[traj_info.traj_idx, 4, :-1]) ** 2
            )
            / self.wheel_base
        )

        v_post = traj_info.xs[traj_info.traj_idx, 3, 1:]
        steer_post = traj_info.xs[traj_info.traj_idx, 4, 1:]
        avu_avp = self._diff_barrier(v_post - self.v_max, self.margin_rate * self.v_max)
        avl_avp = self._diff_barrier(
            -v_post + self.v_min, -self.margin_rate * self.v_min
        )
        asu_asp = self._diff_barrier(
            steer_post - self.steer_max, self.margin_rate * self.steer_max
        )
        asl_asp = self._diff_barrier(
            -steer_post + self.steer_min, -self.margin_rate * self.steer_min
        )
        avu_av = avu_avp * traj_info.fxs[3, 3, :]
        avl_av = -avl_avp * traj_info.fxs[3, 3, :]
        avu_aa = avu_avp * traj_info.fus[3, 0, :]
        avl_aa = -avl_avp * traj_info.fus[3, 0, :]
        asu_as = asu_asp * traj_info.fxs[4, 4, :]
        asl_as = -asl_asp * traj_info.fxs[4, 4, :]
        asu_aw = asu_asp * traj_info.fus[4, 1, :]
        asl_aw = -asl_asp * traj_info.fus[4, 1, :]
        traj_info.fxs[5, 3, :] = avu_av + avl_av
        traj_info.fxs[5, 4, :] = asu_as + asl_as
        traj_info.fus[5, 0, :] = avu_aa + avl_aa
        traj_info.fus[5, 1, :] = asu_aw + asl_aw

    def calc_hessian(
        self,
        traj_info: TrajInfo,
    ) -> None:
        dt_sin_yaw = self.dt * np.sin(traj_info.xs[traj_info.traj_idx, 2, :-1])
        dt_cos_yaw = self.dt * np.cos(traj_info.xs[traj_info.traj_idx, 2, :-1])
        tan_steer = np.tan(traj_info.xs[traj_info.traj_idx, 4, :-1])

        traj_info.fxxs[0, 2, 2, :] = (
            -traj_info.xs[traj_info.traj_idx, 3, :-1] * dt_cos_yaw
        )
        traj_info.fxxs[0, 2, 3, :] = -dt_sin_yaw
        traj_info.fxxs[0, 3, 2, :] = -dt_sin_yaw

        traj_info.fxxs[1, 2, 2, :] = (
            -traj_info.xs[traj_info.traj_idx, 3, :-1] * dt_sin_yaw
        )
        traj_info.fxxs[1, 2, 3, :] = dt_cos_yaw
        traj_info.fxxs[1, 3, 2, :] = dt_cos_yaw

        traj_info.fxxs[2, 4, 4, :] = (
            self.dt
            / self.wheel_base
            * traj_info.xs[traj_info.traj_idx, 3, :-1]
            * (2 * tan_steer**2 + 2)
            * tan_steer
        )
        traj_info.fxxs[2, 3, 4, :] = (
            self.dt
            / self.wheel_base
            * 1
            / (np.cos(traj_info.xs[traj_info.traj_idx, 4, :-1]) ** 2)
        )
        traj_info.fxxs[2, 4, 3, :] = (
            self.dt
            / self.wheel_base
            * 1
            / (np.cos(traj_info.xs[traj_info.traj_idx, 4, :-1]) ** 2)
        )

        v_post = traj_info.xs[traj_info.traj_idx, 3, 1:]
        steer_post = traj_info.xs[traj_info.traj_idx, 4, 1:]
        # C = (v + dt*a) - v_max
        # aB_av = aB_aC*aC_av = aB_aC
        # a2B_a2v = a2B_aC2*aC_av = a2B_aC2
        # C = -(v + dt*a) + v_min
        # aB_av = aB_aC*aC_av = -aB_aC
        # a2B_a2v = -a2B_aC2*aC_av = a2B_aC2
        a2vu_ac2 = self._diff_twice_barrier(v_post - self.v_max, self.margin_rate * self.v_max)
        a2vl_ac2 = self._diff_twice_barrier(
            -v_post + self.v_min, -self.margin_rate * self.v_min
        )
        traj_info.fxxs[5, 3, 3, :] = a2vu_ac2 + a2vl_ac2
        # C = (v + dt*a) - v_max
        # aB_av = aB_aC*aC_av = aB_aC
        # a2B_avaa = a2B_aC2*aC_aa = dt*a2B_aC2
        # C = -(v + dt*a) + v_min
        # aB_av = aB_aC*aC_av = -aB_aC
        # a2B_avaa = -a2B_aC2*aC_aa = dt*a2B_aC2
        traj_info.fxus[5, 3, 0, :] = self.dt*a2vu_ac2 + self.dt*a2vl_ac2
        traj_info.fuus[5, 0, 0, :] = self.dt*self.dt*a2vu_ac2 + self.dt*self.dt*a2vl_ac2
        a2su_ac2 = self._diff_twice_barrier(steer_post - self.steer_max, self.margin_rate * self.steer_max)
        a2sl_ac2 = self._diff_twice_barrier(
            -steer_post + self.steer_min, -self.margin_rate * self.steer_min
        )
        traj_info.fxxs[5, 4, 4, :] = a2su_ac2 + a2sl_ac2
        traj_info.fxus[5, 4, 1, :] = self.dt*a2su_ac2 + self.dt*a2sl_ac2
        traj_info.fuus[5, 1, 1, :] = self.dt*self.dt*a2su_ac2 + self.dt*self.dt*a2sl_ac2


