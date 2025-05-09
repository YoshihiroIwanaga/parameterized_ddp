import numpy as np

from parametric_ddp.utils.TrajInfo import TrajInfo
from problem_setting.abstract.Config import Config


class BackwardPassResult:
    def __init__(self, n: int, m: int, horizon: int) -> None:
        self.ku: np.ndarray = np.zeros((m, horizon))
        self.Ku: np.ndarray = np.zeros((m, n, horizon))
        self.kx0: np.ndarray = np.zeros((n,))
        self.Qu_error: float = 0.0
        self.Qx0_error: float = 0.0
        self.optimal_error: float = 0.0


class MatrixSet:
    def __init__(self) -> None:
        self.value_function_x: np.ndarray = np.zeros((0, 0))
        self.value_function_xx: np.ndarray = np.zeros((0, 0))
        self.Qx: np.ndarray = np.zeros((0, 0))
        self.Qu: np.ndarray = np.zeros((0, 0))
        self.Qxx: np.ndarray = np.zeros((0, 0))
        self.Qxu: np.ndarray = np.zeros((0, 0))
        self.Quu: np.ndarray = np.zeros((0, 0))


class BackwardPass:
    def __init__(self, cfg: Config) -> None:
        self.horizon: int = cfg.horizon
        self.n: int = cfg.n  # state diemnsion
        self.m: int = cfg.m  # input dimension
        self.free_state_idx = (
            cfg.free_state_idx
        )  # indices corresponds to the state which is not fixed at initial time step
        self.failed: bool = False
        self.normalize_Qu_error: bool = cfg.normalize_Qu_error
        if self.normalize_Qu_error:
            self.normalize_factor: float = cfg.Qu_normalization_factor
        else:
            self.normalize_factor: float = 1.0
        self.Qu_error_pre: float = 0.0
        self.Qx0_error_pre: float = 0.0

    def _diff_Q(
        self,
        traj: TrajInfo,
        ms: MatrixSet,
        i: int,
        use_hessian: bool,
    ) -> None:
        ms.Qx = traj.stage_cost_xs[:, :, i] + traj.fxs[:, :, i].T @ ms.value_function_x
        ms.Qu = traj.stage_cost_us[:, :, i] + traj.fus[:, :, i].T @ ms.value_function_x
        quu = traj.stage_cost_uus[:, :, i]
        if use_hessian:
            Vx_reshaped = ms.value_function_x[:, :, np.newaxis]
            ms.Qxx = (
                traj.stage_cost_xxs[:, :, i]
                + traj.fxs[:, :, i].T @ ms.value_function_xx @ traj.fxs[:, :, i]
                + np.sum(Vx_reshaped * traj.fxxs[:, :, :, i], axis=0)
            )
            ms.Qxu = (
                traj.stage_cost_xus[:, :, i]
                + traj.fxs[:, :, i].T @ ms.value_function_xx @ traj.fus[:, :, i]
                + np.sum(Vx_reshaped * traj.fxus[:, :, :, i], axis=0)
            )
            ms.Quu = (
                quu
                + traj.fus[:, :, i].T @ ms.value_function_xx @ traj.fus[:, :, i]
                + np.sum(Vx_reshaped * traj.fuus[:, :, :, i], axis=0)
            )
        else:
            ms.Qxx = (
                traj.stage_cost_xxs[:, :, i]
                + traj.fxs[:, :, i].T @ ms.value_function_xx @ traj.fxs[:, :, i]
            )
            ms.Qxu = (
                traj.stage_cost_xus[:, :, i]
                + traj.fxs[:, :, i].T @ ms.value_function_xx @ traj.fus[:, :, i]
            )
            ms.Quu = (
                quu + traj.fus[:, :, i].T @ ms.value_function_xx @ traj.fus[:, :, i]
            )

    def _calc_feedforward_and_feedback(
        self, result: BackwardPassResult, ms: MatrixSet, i: int
    ) -> None:
        if i == 0 and len(self.free_state_idx) > 0:
            Qxfree = ms.Qx[self.free_state_idx, :]  # type:ignore
            Qxfreexfree = ms.Qxx[
                np.ix_(self.free_state_idx, self.free_state_idx)  # type:ignore
            ]
            Qxfreeu = ms.Qxu[self.free_state_idx, :]
            A_ = np.vstack(
                (
                    np.hstack((ms.Quu, Qxfreeu.T)),
                    np.hstack((Qxfreeu, Qxfreexfree)),
                )
            )
            try:
                x_ = np.linalg.solve(A_, -np.vstack((ms.Qu, Qxfree)))
            except np.linalg.LinAlgError:
                self.failed = True
                return
            result.ku[:, i : i + 1] = x_[: self.m, :]
            result.kx0[self.free_state_idx,] = x_[
                self.m : (self.m + len(self.free_state_idx)),
                0,
            ]
        else:
            try:
                x_ = np.linalg.solve(
                    ms.Quu,
                    -np.hstack((ms.Qu, ms.Qxu.T)),
                )
            except np.linalg.LinAlgError:
                self.failed = True
                return
            result.ku[:, i : i + 1] = x_[:, 0:1]
            result.Ku[:, :, i] = x_[:, 1:]

    def _update_V_diff(self, result: BackwardPassResult, ms: MatrixSet, i: int) -> None:
        ms.value_function_x = (
            ms.Qx
            + result.Ku[:, :, i].T @ ms.Qu
            + result.Ku[:, :, i].T @ ms.Quu @ result.ku[:, i : i + 1]
            + ms.Qxu @ result.ku[:, i : i + 1]
        )
        ms.value_function_xx = (
            ms.Qxx
            + result.Ku[:, :, i].T @ ms.Qxu.T
            + ms.Qxu @ result.Ku[:, :, i]
            + result.Ku[:, :, i].T @ ms.Quu @ result.Ku[:, :, i]
        )
        ms.value_function_xx = 1 / 2 * (ms.value_function_xx + ms.value_function_xx.T)

    def _update_errors(self, result: BackwardPassResult, ms: MatrixSet, i: int) -> None:
        result.Qu_error = max(
            [result.Qu_error, self.normalize_factor * np.max(np.abs(ms.Qu))]
        )
        if i == 0 and len(self.free_state_idx) > 0:
            result.Qx0_error = self.normalize_factor * np.max(
                np.abs(ms.Qx[self.free_state_idx, :])
            )

    def update(
        self,
        traj: TrajInfo,
        use_hessian: bool = False
    ) -> BackwardPassResult:
        result = BackwardPassResult(self.n, self.m, self.horizon)
        matrix_set = MatrixSet()
        matrix_set.value_function_x = traj.terminal_cost_x.reshape((self.n, 1))
        matrix_set.value_function_xx = traj.terminal_cost_xx

        for i in reversed(range(0, self.horizon)):
            self._diff_Q(traj, matrix_set, i, use_hessian)
            self._calc_feedforward_and_feedback(result, matrix_set, i)
            if self.failed:
                result.Qu_error = self.Qu_error_pre
                result.Qx0_error = self.Qx0_error_pre
                return result
            self._update_errors(result, matrix_set, i)
            if i != 0:
                self._update_V_diff(result, matrix_set, i)

        result.optimal_error = max(
            [
                result.Qu_error,
                result.Qx0_error,
            ]
        )
        self.Qu_error_pre = result.Qu_error
        self.Qx0_error_pre = result.Qx0_error
        return result
