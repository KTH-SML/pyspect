import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets


class DriftingCanoe(dynamics.ControlAndDisturbanceAffineDynamics):
    """Canoe carried downstream at unit speed, steering sideways only."""

    def __init__(self,
                 uMax=+1.0,
                 uMin=-1.0,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):
        self.uMax = uMax
        self.uMin = uMin
        if control_space is None:
            control_space = sets.Box(jnp.array([uMin]), jnp.array([uMax]))
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([0.0]), jnp.array([0.0]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def with_mode(self, mode: str):
        assert mode in ["reach", "avoid"]
        if mode == "reach":
            self.control_mode = "min"
            self.disturbance_mode = "max"
        elif mode == "avoid":
            self.control_mode = "max"
            self.disturbance_mode = "min"
        return self

    def open_loop_dynamics(self, state, time):
        return jnp.array([0.0, -1.0])

    def control_jacobian(self, state, time):
        return jnp.array([[1.0], [0.0]])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[0.0], [0.0]])


class DriftingCanoeBall(DriftingCanoe):
    """Same drift, but the paddler can push in any direction at unit speed."""

    def __init__(self,
                 uMax=+1.0,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):
        if control_space is None:
            control_space = sets.Ball(jnp.array([0.0, 0.0]), uMax)
        super().__init__(uMax=uMax,
                         control_mode=control_mode,
                         disturbance_mode=disturbance_mode,
                         control_space=control_space,
                         disturbance_space=disturbance_space)

    def control_jacobian(self, state, time):
        return jnp.array([[1.0, 0.0], [0.0, 1.0]])
