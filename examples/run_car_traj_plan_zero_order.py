import math

import numpy as np

from parametric_ddp.ddp_solver.DdpSolver import DdpSolver
from parametric_ddp.interpolator.zero_order.ZeroOrderHolderConfigWrapper import (
    ZeroOrderHolderConfigWrapper,
)
from parametric_ddp.interpolator.zero_order.ZeroOrderHolderWrapper import (
    ZeroOrderHolderWrapper,
)
from parametric_ddp.utils.OptimalControlProblem import OptimalControlProblem
from parametric_ddp.utils.TrajInfo import TrajInfo
from parametric_ddp.utils.TrajInfoWrapper import TrajInfoWrapper
from problem_setting.car.CarConfig import CarConfig
from problem_setting.car.CarCost import CarCost
from problem_setting.car.CarDynamics import CarDynamics
from problem_setting.car.utils.plot_car_lib import create_gif
from utils.plot_traj_lib import plot_us, plot_xs

if __name__ == "__main__":
    # problem setting
    x_ref = np.array([2.5, -3.0, math.pi, 0, 0, 0])
    knot_step = 50
    steps_per_knot = 20 * [knot_step]
    cfg = CarConfig(x_ref=x_ref, horizon=sum(steps_per_knot))
    dynamics = CarDynamics(cfg)

    # solve with original IP-iLQR
    traj_info = TrajInfo(cfg, cfg.step_size_num)
    cost = CarCost(cfg)
    ocp = OptimalControlProblem(dynamics, cost, cfg)
    solver = DdpSolver(ocp, traj_info, cfg)
    solver.run()
    xs_opt, us_opt = solver.get_optimal_traj()

    # solve with zero-order interpolated P-IP-iLQR
    cfg_zero_order = ZeroOrderHolderConfigWrapper(
        cfg,
        steps_per_knot,
    )
    traj_info_zero_order = TrajInfoWrapper(cfg_zero_order, cfg.step_size_num)
    cost_zero_order = CarCost(cfg_zero_order)
    ocp_zero_order = ZeroOrderHolderWrapper(dynamics, cost_zero_order, cfg_zero_order)
    solver_zero_order = DdpSolver(ocp_zero_order, traj_info_zero_order, cfg_zero_order)
    solver_zero_order.run()
    xs_opt_zero_order, us_opt_zero_order = solver_zero_order.get_optimal_traj()

    # plot
    plot_xs(
        xs_opt,
        xs_opt_zero_order,
        cfg.x_ref,
        steps_per_knot,
        cfg_zero_order.time_step_on_knot,
        cfg.horizon,
    )

    plot_us(
        us_opt,
        us_opt_zero_order,
        steps_per_knot,
        cfg_zero_order.time_step_on_knot,
        cfg.horizon,
    )

    # eval comp time
    diff_time = np.array(solver.comp_times.diff_times)
    fp_time = np.array(solver.comp_times.fp_times)
    bp_time = np.array(solver.comp_times.bp_times)
    total = diff_time + fp_time + bp_time
    if cfg.run_ddp:
        print("Original DDP took : ", np.sum(total), "[sec]")
    else:
        print("Original iLQR took : ", np.sum(total), "[sec]")
    comp_time = [0]
    tmp = 0
    for t in total:
        tmp += t
        comp_time.append(tmp)

    diff_time = np.array(solver_zero_order.comp_times.diff_times)
    fp_time = np.array(solver_zero_order.comp_times.fp_times)
    bp_time = np.array(solver_zero_order.comp_times.bp_times)
    total = diff_time + fp_time + bp_time
    if cfg.run_ddp:
        print("Zero-Order Interpolated DDP took : ", np.sum(total), "[sec]")
    else:
        print("Zero-Order Interpolated iLQR took : ", np.sum(total), "[sec]")
    comp_time_zero_order = [0]
    tmp = 0
    for t in total:
        tmp += t
        comp_time_zero_order.append(tmp)

    # crearte gif
    ts = np.linspace(0.0, max(comp_time[-1], comp_time_zero_order[-1]), 50)
    xs = []
    us = []
    interpolated_xs = []
    interpolated_us = []
    for t_ in ts:
        index = np.abs(np.array(comp_time) - t_).argmin()
        index_zero_order = np.abs(np.array(comp_time_zero_order) - t_).argmin()
        traj = solver.history.traj[index]
        traj_zero_order = solver_zero_order.history.traj[index_zero_order]
        xs.append(traj.xs[traj.traj_idx, :, :])
        us.append(traj.us[traj.traj_idx, :, :])
        interpolated_xs.append(
            traj_zero_order.every_traj.xs[traj_zero_order.every_traj.traj_idx, :, :]
        )
        interpolated_us.append(
            traj_zero_order.every_traj.us[traj_zero_order.every_traj.traj_idx, :, :]
        )

    create_gif(
        "./car_traj_plan_zero_order.gif",
        xs,
        us,
        interpolated_xs,
        interpolated_us,
        xs_opt,
        cfg.horizon,
        cfg.x_ref,
        3.0,
    )
