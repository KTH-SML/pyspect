import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets


class Bicycle5D(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 min_steer, max_steer,
                 min_accel, max_accel,
                 min_disturbances=None, 
                 max_disturbances=None,
                 wheelbase=0.32,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):

        self.wheelbase = wheelbase

        if min_disturbances is None:
            min_disturbances = [0] * 5
        if max_disturbances is None:
            max_disturbances = [0] * 5

        if control_space is None:
            control_space = sets.Box(jnp.array([min_steer, min_accel]),
                                     jnp.array([max_steer, max_accel]))
        
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

    ## Dynamics
    # Implements the affine dynamics 
    #   `dx_dt = f(x, t) + G_u(x, t) @ u + G_d(x, t) @ d`.
    # 
    #   x_dot = v * cos(yaw) + d_1
    #   y_dot = v * sin(yaw) + d_2
    #   yaw_dot = (v * tan(delta))/L + d3
    #   delta_dot = u1 + d4
    #   v_dot = u2 + d5
        

    def open_loop_dynamics(self, state, time):
        x, y, yaw, delta, vel = state
        return jnp.array([
            vel * jnp.cos(yaw),
            vel * jnp.sin(yaw),
            (vel * jnp.tan(delta))/self.wheelbase,
            0.,
            0.,
        ])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [1., 0.],
            [0., 1.],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.identity(5)
