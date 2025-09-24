import jax.numpy as jnp

from .. import dynamics
from .. import sets


class ElectricVehicle(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 min_accel, max_accel,
                 tau=20,
                 gam=2e-3,
                 min_disturbances=None, 
                 max_disturbances=None,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):

        # v̇ = −v/τ
        # ḃ = −γ( v a + v²/τ )
        self.tau = tau
        self.gam = gam

        if min_disturbances is None:
            min_disturbances = [0] * 3
        if max_disturbances is None:
            max_disturbances = [0] * 3

        if control_space is None:
            assert isinstance(min_accel, (int, float)), 'min_accel must be a number'
            assert isinstance(max_accel, (int, float)), 'max_accel must be a number'
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
        x, v, b = state
        return jnp.array([
                                        v,
                          -1/self.tau * v,
            -self.gam * 1/self.tau * v**2,
        ])

    def control_jacobian(self, state, time):
        x, v, b = state
        return jnp.array([
            [          0.0],
            [          1.0],
            [-self.gam * v],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.identity(3)
