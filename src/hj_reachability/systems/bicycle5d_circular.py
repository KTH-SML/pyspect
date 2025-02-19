import jax.numpy as jnp

from .. import dynamics
from .. import sets


class Bicycle5DCircular(dynamics.Dynamics):

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

    def __call__(self, state, control, disturbance, time):
        (r,      # 0
         phi,    # 1
         v_r,    # 2
         v_phi,  # 3
         v_yaw,  # 4
        ) = state
        (acc,    # 0
         delta,  # 1
        ) = control
        return jnp.array([
            v_r,
            v_phi,
            v_yaw * v_phi,
            acc - v_yaw * v_r,
            v_phi * jnp.tan(delta) / self.wheelbase,
        ])
    
    def optimal_control_and_disturbance(self, state, time, grad_value):
        # OBS: only when vel >= 0

        # if self.control_mode == 'max':
        #     opt_ctrl_acc  = jnp.where(0 <= grad_value[3],
        #                               self.control_space.hi[0],
        #                               self.control_space.lo[0])
        #     opt_ctrl_vyaw = jnp.where(0 <= jnp.hypot(grad_value[2], grad_value[3]),
        #                               self.control_space.hi[1],
        #                               self.control_space.lo[1])
        #     opt_ctrl = jnp.array([opt_ctrl_acc, opt_ctrl_vyaw])
        # else:
        #     opt_ctrl_acc = jnp.where(0 <= grad_value[3],
        #                              self.control_space.lo[0],
        #                              self.control_space.hi[0])
        #     opt_ctrl_vyaw = jnp.where(0 <= jnp.hypot(grad_value[2], grad_value[3]),
        #                               self.control_space.hi[1],
        #                               self.control_space.lo[1])
        #     opt_ctrl = jnp.array([opt_ctrl_acc, opt_ctrl_vyaw])

        if self.control_mode == 'max':
            opt_ctrl  = jnp.where(0 <= grad_value[3:], self.control_space.hi, self.control_space.lo)
        else:
            opt_ctrl  = jnp.where(0 <= grad_value[3:], self.control_space.lo, self.control_space.hi)

        if self.disturbance_mode == 'max':
            opt_dstb = jnp.where(0 <= grad_value, self.disturbance_space.hi, self.disturbance_space.lo)
        else:
            opt_dstb = jnp.where(0 <= grad_value, self.disturbance_space.lo, self.disturbance_space.hi)

        return opt_ctrl, opt_dstb
    
    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        del value, grad_value_box # unused
        return jnp.abs(self(state, self.control_space.max_magnitudes, self.disturbance_space.max_magnitudes, time))
