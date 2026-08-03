import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets


class Canoe(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 uMax=1,
                 dMax=1,
                 dMin=0,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):
        self.uMax = uMax
        self.dMax = dMax
        self.dMin = dMin
        if control_space is None:
            control_space = sets.Ball(jnp.array([0, 0]), uMax)
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([self.dMin]), jnp.array([self.dMax]))
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
        return jnp.array([0.0, 0.0])

    def control_jacobian(self, state, time):
        return jnp.array([[1.0, 0.0], [0.0, 1.0]])

    def disturbance_jacobian(self, state, time):
        x, y = state
        return jnp.array([[jnp.minimum(x/4.0, 0.0*x)], [0.0]])
