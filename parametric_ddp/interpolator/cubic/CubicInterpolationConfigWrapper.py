import numpy as np

from problem_setting.abstract.Config import Config


class CubicInterpolationConfigWrapper(Config):
    def __init__(
        self,
        cfg: Config,
        steps_per_knot,
    ) -> None:
        self.org_cfg: Config = cfg
        self.n: int = cfg.n + cfg.m + cfg.m
        self.m: int = cfg.m
        self.dt: float = cfg.dt
        self.horizon = len(steps_per_knot)
        self.x_ini: np.ndarray = np.hstack(
            (
                cfg.x_ini,
                np.zeros(cfg.m),
                np.zeros(cfg.m),
            )
        )
        self.x_ref: np.ndarray = np.hstack(
            (
                cfg.x_ref,
                np.zeros(cfg.m),
                np.zeros(cfg.m),
            )
        )
        self.Q_ini = np.diag(
            np.hstack((np.diag(cfg.Q), 0.5 * np.diag(cfg.R), 0 * np.diag(cfg.R)))
        )
        self.Q: np.ndarray = np.diag(
            np.hstack((np.diag(cfg.Q), np.diag(cfg.R), 0 * np.diag(cfg.R)))
        )
        self.R: np.ndarray = 0.1 * cfg.R
        self.Q_terminal: np.ndarray = np.diag(
            np.hstack(
                (
                    np.diag(cfg.Q_terminal),
                    0.5 * steps_per_knot[0] * np.diag(cfg.R),
                    0.5 * steps_per_knot[0] * 0 * np.diag(cfg.R),
                )
            )
        )
        self.free_state_idx = cfg.free_state_idx + list(range(cfg.n, cfg.n + 2 * cfg.m))
        self.tol = cfg.tol
        self.max_iter: int = cfg.max_iter
        self.run_ddp: bool = cfg.run_ddp
        self.step_size_num: int = cfg.step_size_num
        self.cost_change_ratio_convergence_criteria = (
            cfg.cost_change_ratio_convergence_criteria
        )
        self.steps_per_knot = steps_per_knot  # [L_0, L_1,..., L_{Te-1}] ∈ N^{Te}
        self.is_step_unique: bool = all(
            val == self.steps_per_knot[0] for val in self.steps_per_knot
        )
        self.time_step_on_knot = [0]  # [k_(0), k_(1),..., k_(Te)] ∈ N^{Te+1}
        input_update_step = 0
        for step in self.steps_per_knot:
            input_update_step += step
            self.time_step_on_knot.append(input_update_step)
        knot_idx = 0
        time_step_from_knot = 0
        self.knot_idxs = []  # [0,..,0,1,...,1,...,Te,...,Te] ∈ N^{te}
        self.time_step_from_knots = (
            []
        )  # [0,1..,L_0-1,0,1...,L_1-1,...,0,1,...,L_{Te}-1] ∈ N^{te}
        for i in range(cfg.horizon):
            if i == self.time_step_on_knot[knot_idx]:
                knot_idx += 1
                time_step_from_knot = 0
            self.knot_idxs.append(knot_idx)
            self.time_step_from_knots.append(time_step_from_knot)
            time_step_from_knot += 1
        self.u_inis = np.zeros((self.step_size_num, self.m, self.horizon))
        self.normalize_Qu_error = True
        self.Qu_normalization_factor = 1.0 / self.steps_per_knot[0]
        self.knows_optimal_cost = False
        self.optimal_cost = 0.0
