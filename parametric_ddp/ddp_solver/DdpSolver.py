import copy
import typing
from resource import RUSAGE_SELF
from resource import getrusage as resource_usage
from time import time as timestamp

import numpy as np

from parametric_ddp.ddp_solver.BackwardPass import BackwardPass
from parametric_ddp.ddp_solver.ForwardPass import ForwardPass
from parametric_ddp.utils.OptimalControlProblem import OptimalControlProblem
from parametric_ddp.utils.TrajInfo import TrajInfo
from parametric_ddp.utils.TrajInfoWrapper import TrajInfoWrapper
from problem_setting.abstract.Config import Config


class OptimizationHistory:
    def __init__(self) -> None:
        self.traj: list[np.ndarray] = []
        self.barrier_param: list[float] = []
        self.step_size: list[float] = []
        self.opt_error: list[float] = []
        self.Qu_error: list[float] = []
        self.Qx0_error: list[float] = []
        self.gain_u: list[np.ndarray] = []
        self.ff_u: list[np.ndarray] = []
        self.state: list[np.ndarray] = []
        self.input: list[np.ndarray] = []


class ComputationalTimes:
    def __init__(self) -> None:
        self.traj_initialization_time: float = 0.0
        self.diff_times: list[float] = []
        self.fp_times: list[float] = []
        self.bp_times: list[float] = []


class DdpSolver:
    def __init__(
        self,
        ocp: OptimalControlProblem,
        traj: typing.Union[TrajInfo, TrajInfoWrapper],
        cfg: Config,
    ) -> None:
        self.ocp: OptimalControlProblem = ocp
        self.fp: ForwardPass = ForwardPass(ocp, cfg)
        self.bp: BackwardPass = BackwardPass(cfg)
        self.traj: typing.Union[TrajInfo, TrajInfoWrapper] = traj
        self.traj_pre: typing.Union[TrajInfo, TrajInfoWrapper] = copy.deepcopy(traj)
        self.history: OptimizationHistory = OptimizationHistory()
        self.comp_times: ComputationalTimes = ComputationalTimes()
        self.ocp.set_constant(self.traj)
        self.horizon: int = cfg.horizon
        self.tol: float = cfg.tol
        self.max_iter: int = cfg.max_iter
        self.run_ddp: bool = cfg.run_ddp
        self.is_warm_start_enabled: bool = False
        self.measures_time: bool = True
        self.prints_info: bool = False
        self.cost_change_ratio_convergence_criteria: float = (
            cfg.cost_change_ratio_convergence_criteria
        )
        self.knows_optmial_cost: bool = cfg.knows_optimal_cost
        self.given_optmial_cost: float = cfg.optimal_cost
        self.fp_done_cnt: int = 0
        self.start_ddp_cnt: int = 15

    def _preprocess(self) -> None:
        if self.measures_time:
            _, start_resources = timestamp(), resource_usage(RUSAGE_SELF)

        for t in range(self.horizon):
            self.ocp.transit(self.traj, t)
        self.ocp.calc_cost(self.traj)

        if self.is_warm_start_enabled:
            self.traj.set_traj_idx(np.nanargmin(self.traj.costs))  # type:ignore
        else:
            self.traj.set_traj_idx(0)

        self.traj_pre = copy.deepcopy(self.traj)

        if self.measures_time:
            end_resources, _ = resource_usage(RUSAGE_SELF), timestamp()
            self.comp_times.traj_initialization_time = (
                end_resources.ru_utime - start_resources.ru_utime  # type:ignore
            )

        if self.prints_info:
            print(
                "initial : ",
                "cost : ",
                self.traj.costs[self.traj.traj_idx],
            )
        self.history.traj.append(copy.deepcopy(self.traj))  # type:ignore

    def _start_timer(self) -> None:
        if self.measures_time:
            _, self.start_resources = timestamp(), resource_usage(RUSAGE_SELF)

    def _stop_timer(self) -> float:
        if self.measures_time:
            end_resources, _ = resource_usage(RUSAGE_SELF), timestamp()
            comp_time = end_resources.ru_utime - self.start_resources.ru_utime
        else:
            comp_time = 0.0
        return comp_time

    def _eval_terminate_condition(
        self, iter: int, optimal_error: float, cost_change_ratio: float
    ) -> bool:
        terminate_iter = False
        if self.fp.failed:
            self.terminate_condition = "Failed Forward Pass"
            terminate_iter = True
            return terminate_iter

        if self.bp.failed:
            self.terminate_condition = "Failed Backward Pass"
            terminate_iter = True
            return terminate_iter

        if optimal_error <= self.tol:
            self.terminate_condition = "Optimal Error is less that tolerance"
            terminate_iter = True
            return terminate_iter

        if (
            cost_change_ratio  # type:ignore
            > self.cost_change_ratio_convergence_criteria
        ):
            self.terminate_condition = "cost do not change anymore"
            terminate_iter = True
            return terminate_iter

        if self.knows_optmial_cost:
            if 0.999 * self.traj.costs[self.traj.traj_idx] < self.given_optmial_cost:
                self.terminate_condition = "Reach the Given Optimal Cost"
                terminate_iter = True
                return terminate_iter

        if iter == self.max_iter - 1:
            self.terminate_condition = "Reached Max Iteration Number"
            terminate_iter = True
            return terminate_iter

        return terminate_iter

    def _iterate(self) -> None:
        for iter in range(self.max_iter):
            use_hessian = self.run_ddp and (self.fp_done_cnt > self.start_ddp_cnt)
            # Differetiation Part
            self._start_timer()
            self.ocp.diff(self.traj, calc_dynamics_hessian=use_hessian)
            self.comp_times.diff_times.append(self._stop_timer())

            # Backward Pass
            self._start_timer()
            bp_result = self.bp.update(self.traj, use_hessian=use_hessian)
            self.comp_times.bp_times.append(self._stop_timer())

            # Forward Pass
            self._start_timer()
            if not self.bp.failed:
                self.fp.update(
                    self.traj,
                    self.traj_pre,
                    bp_result,
                )
                if self.fp.failed:
                    self.traj = copy.deepcopy(self.traj_pre)
                    cost_change_ratio = 0.0
                else:
                    cost_change_ratio = np.abs(
                        self.traj.costs[self.traj.traj_idx]
                        / self.traj_pre.costs[self.traj_pre.traj_idx]
                    )
                    self.traj_pre = copy.deepcopy(self.traj)
                    self.fp_done_cnt += 1
            self.comp_times.fp_times.append(self._stop_timer())

            if self.prints_info:
                print(
                    "iter : ",
                    iter,
                    "cost : ",
                    self.traj.costs[self.traj.traj_idx],
                )

            self.history.traj.append(copy.deepcopy(self.traj))  # type:ignore
            self.history.Qu_error.append(bp_result.Qu_error)
            self.history.Qx0_error.append(bp_result.Qx0_error)
            self.history.step_size.append(self.fp.step_sizes[self.traj.traj_idx][0])

            if self._eval_terminate_condition(
                iter, bp_result.optimal_error, cost_change_ratio
            ):
                break

    def get_optimal_traj(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        if self.traj.every_traj is None:
            return (
                self.traj.xs[self.traj.traj_idx, :, :],
                self.traj.us[self.traj.traj_idx, :, :],
            )
        else:
            return (
                self.traj.every_traj.xs[self.traj.traj_idx, :, :],
                self.traj.every_traj.us[self.traj.traj_idx, :, :],
            )

    def run(self) -> None:
        self._preprocess()
        self._iterate()
