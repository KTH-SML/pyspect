import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets


class SingleIntegrator(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 min_vel, max_vel,
                 ndim=1,
                 min_disturbances=None, 
                 max_disturbances=None,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):

        self._open_loop_dynamics = jnp.kron(
            jnp.array([
                [0.],
            ]),
            jnp.eye(ndim)
        )

        self._control_jacobian = jnp.kron(
            jnp.array([
                [1.],
            ]),
            jnp.eye(ndim)
        )

        self._disturbance_jacobian = jnp.kron(
            jnp.identity(1),
            jnp.eye(ndim)
        )

        if min_disturbances is None:
            min_disturbances = [0] * (1*ndim)
        if max_disturbances is None:
            max_disturbances = [0] * (1*ndim)

        if control_space is None:
            control_space = sets.Box(jnp.array([min_vel]),
                                     jnp.array([max_vel]))
        
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
        return self._open_loop_dynamics @ state

    def control_jacobian(self, state, time):
        return self._control_jacobian

    def disturbance_jacobian(self, state, time):
        return self._disturbance_jacobian
