import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets


class PKPD(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 uMax=1.0,
                 uMin=0.0,
                 dMax=1.0,
                 dMin=1.0,
                 gamma=0.0,
                 delta=0.3,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin
        self.gamma = gamma
        self.delta = delta
        if control_space is None:
            control_space = sets.Box(jnp.array([uMin]), jnp.array([uMax]))
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([dMin]), jnp.array([dMax]))
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
        x1, x2, x3 = state
        return jnp.array([
            0.5 * x2**4 / (x2**4 + 0.5**4) - self.gamma * x1,
            -2 * x2,
            x2 - self.delta * x3,
        ])

    def control_jacobian(self, state, time):
        return jnp.array([[0.0], [1.0], [0.0]])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[0.0], [0.0], [0.0]])
