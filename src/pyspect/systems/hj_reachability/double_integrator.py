import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets


class DoubleIntegrator(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 min_accel, max_accel,
                 min_disturbances=None, 
                 max_disturbances=None,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):

        if min_disturbances is None:
            min_disturbances = [0] * 2
        if max_disturbances is None:
            max_disturbances = [0] * 2

        if control_space is None:
            control_space = sets.Box(jnp.array([min_accel]),
                                     jnp.array([max_accel]))
        
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array(min_disturbances),
                                         jnp.array(max_disturbances))
        super().__init__(control_mode, 
                         disturbance_mode, 
                         control_space, 
                         disturbance_space)

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
        return jnp.array([
            [0., 1.],
            [0., 0.],
        ]) @ state

    def control_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [1.],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.identity(2)
