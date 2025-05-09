import numpy as np

from parametric_ddp.ddp_solver.DdpSolver import DdpSolver
from parametric_ddp.interpolator.linear.LinearInterpolationConfigWrapper import (
    LinearInterpolationConfigWrapper,
)
from parametric_ddp.interpolator.linear.LinearInterpolationWrapper import (
    LinearInterpolationWrapper,
)
from parametric_ddp.utils.OptimalControlProblem import OptimalControlProblem
from parametric_ddp.utils.TrajInfo import TrajInfo
from parametric_ddp.utils.TrajInfoWrapper import TrajInfoWrapper
from problem_setting.inverted_pendulum.InvertedPendulumConfig import (
    InvertedPendulumConfig,
)
from problem_setting.inverted_pendulum.InvertedPendulumCost import InvertedPendulumCost
from problem_setting.inverted_pendulum.InvertedPendulumDynamics import (
    InvertedPendulumDynamics,
)
from utils.plot_traj_lib import plot_us, plot_xs

if __name__ == "__main__":
    # problem setting
    knot_step = 20
    steps_per_knot = 25 * [knot_step]
    cfg = InvertedPendulumConfig(horizon=sum(steps_per_knot))
    dynamics = InvertedPendulumDynamics(cfg)

    # solve with original IP-iLQR
    traj_info = TrajInfo(cfg, cfg.step_size_num)
    cost = InvertedPendulumCost(cfg)
    ocp = OptimalControlProblem(dynamics, cost, cfg)
    solver = DdpSolver(ocp, traj_info, cfg)
    solver.run()
    xs_opt, us_opt = solver.get_optimal_traj()

    # solve with linear interpolated P-IP-iLQR
    cfg_linear = LinearInterpolationConfigWrapper(
        cfg,
        steps_per_knot,
    )
    cost_linear = InvertedPendulumCost(cfg_linear)
    traj_info_linear = TrajInfoWrapper(cfg_linear, cfg.step_size_num)
    ocp_linear = LinearInterpolationWrapper(dynamics, cost_linear, cfg_linear)
    solver_linear = DdpSolver(ocp_linear, traj_info_linear, cfg_linear)
    solver_linear.run()
    xs_opt_linear, us_opt_linear = solver_linear.get_optimal_traj()

    # plot
    plot_xs(
        xs_opt,
        xs_opt_linear,
        cfg.x_ref,
        steps_per_knot,
        cfg_linear.time_step_on_knot,
        cfg.horizon,
    )

    plot_us(
        us_opt,
        us_opt_linear,
        steps_per_knot,
        cfg_linear.time_step_on_knot,
        cfg.horizon,
    )

    diff_time = np.array(solver.comp_times.diff_times)
    fp_time = np.array(solver.comp_times.fp_times)
    bp_time = np.array(solver.comp_times.bp_times)
    total = diff_time + fp_time + bp_time
    print("Original iLQR took : ", np.sum(total), "[sec]")
    comp_time = [0]
    tmp = 0
    for t in total:
        tmp += t
        comp_time.append(tmp)

    diff_time = np.array(solver_linear.comp_times.diff_times)
    fp_time = np.array(solver_linear.comp_times.fp_times)
    bp_time = np.array(solver_linear.comp_times.bp_times)
    total = diff_time + fp_time + bp_time
    print("Linear Interpolated iLQR took : ", np.sum(total), "[sec]")
    comp_time_linear = [0]
    tmp = 0
    for t in total:
        tmp += t
        comp_time_linear.append(tmp)

    ts = np.linspace(0.0, max(comp_time[-1], comp_time_linear[-1]), 75)
