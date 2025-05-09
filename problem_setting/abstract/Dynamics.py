from parametric_ddp.utils.TrajInfo import TrajInfo
from problem_setting.abstract.Config import Config


class Dynamics(object):
    def __init__(self, config: Config) -> None:
        self.n: int = config.n
        self.m: int = config.m
        self.dt: float = config.dt

    def set_constant(
        self,
        traj_info: TrajInfo,
    ) -> None:
        pass

    def transit(
        self,
        traj_info: TrajInfo,
        t: int,  # time step
    ) -> None:
        pass

    def calc_jacobian(
        self,
        traj_info: TrajInfo,
    ) -> None:
        pass

    def calc_hessian(
        self,
        traj_info: TrajInfo,
    ) -> None:
        pass
