import heterocl as hcl
import numpy as np
import time

from . import math as hcl_math
from .derivatives import spatial_derivative
from .grid import Grid
from .shapes import *

class Solver: 

    debug = False
    interactive = False
    
    accuracy = 'low'
    
    _executable = None

    def __init__(self, grid, model, *, 
                 interactive=True,
                 debug=False,
                 accuracy='low',
                 dtype=hcl.Float()):

        # Solver options
        self.interactive = interactive
        self.debug = debug

        if self.interactive:
            print("== Welcome to optimized_dp ==")

        # Initialize the HCL environment
        hcl.init(hcl.Float(32))

        self.accuracy = accuracy
        assert self.accuracy == 'low', 'This modification to odp only supports low accuracy'

        self.dtype = dtype

        self.grid = grid
        self.model = model
    

        self.state_shape = (self.model.state_dims,)
        self.ctrl_shape = (self.model.ctrl_dims,)
        self.dstb_shape = (self.model.dstb_dims,)

        self.build()
        
    def __call__(self, *args):
        """
        Run the solver.
        
        This method is intended to be overloaded.
        """

        # Run the executable
        self._executable(*args)
        
    def build(self):

        if self.interactive:
            print('Building...')

        beta = hcl.placeholder((1,), name='beta', dtype=self.dtype)
        vf = hcl.placeholder(self.grid.shape, name="vf", dtype=self.dtype)
        t = hcl.placeholder((2,), name="t", dtype=self.dtype)
        xs = [hcl.placeholder((axlen,), dtype=self.dtype, name=f'x_{i}')
              for i, axlen in enumerate(self.grid.shape)]
        args = [beta, vf, t, *xs]

        if self.debug:
            h = hcl.placeholder(self.grid.shape, name='h', dtype=self.dtype)
            args = [h] + args

        # lambda is necessary so that hcl can modify properties of the function object
        program = lambda *args: self.entrypoint(*args)
        self._sched = hcl.create_schedule(args, program)

        if not self.debug:

            # Accessing the hamiltonian and dissipation stage
            stage_hamiltonian = program.Hamiltonian
            stage_dissipation = program.Dissipation

            # PROBABLY NEED TO ALSO ADD THE VALUE FUNCTION UPDATE AND POTENTIAL SET CHECK HERE

            stage_value_function = program.Value_Function
# THIS STUFF IS COMMENTED!!!!!!!!!!!
   #         stage_potential_update = program.Potential_Update
            # Thread parallelize hamiltonian and dissipation computation
            self._sched[stage_hamiltonian].parallel(stage_hamiltonian.axis[0])
            self._sched[stage_dissipation].parallel(stage_dissipation.axis[0])

 #           self._sched[stage_potential_update].parallel(stage_potential_update.axis[0])
            self._sched[stage_value_function].parallel(stage_value_function.axis[0])


        self._executable = hcl.build(self._sched)

        if self.interactive:
            print(f'> {type(self).__name__} built!\n')

    def entrypoint(self, *args):

        if self.debug:
            h_dbg, beta, vf, t, *xs = args
        else:
            beta, vf, t, *xs = args

        # PLAN:
        # I = immediately updated set, P = potentially updated set
        # Update vf for states in I
        # Simulataneously calculate the new P
        # For all states in P check if they should be updated (changing P)
        # Update those states in the final P
        # Do the dissipation stage for both I and P and time calculation

        # We will need to update eps for P calculation
        # For Hamiltonian function, need to specify the stage of calculation: P or I


        # Initialize intermediate tensors
        p = hcl.compute(self.grid.shape, lambda *_: 0, name="p")
        sum_exp = hcl.compute(self.grid.shape, lambda *_: 0, name="sum_exp")
        num_exp = hcl.compute(self.grid.shape, lambda *_: 0, name="num_exp")
        dv_diff = hcl.compute(self.grid.shape + self.state_shape, lambda *_: 0, name='dv_diff')
        dv_max = hcl.compute(self.state_shape, lambda _: -1e9, name='dv_max')
        dv_min = hcl.compute(self.state_shape, lambda _: +1e9, name='dv_min')
        max_alpha = hcl.compute(self.state_shape, lambda _: -1e9, name='max_alpha')

       
        # vf cut
        hcl.update(vf, lambda *idxs: hcl_math.min(beta.v + 0.1, vf[idxs]))
        hcl.update(vf, lambda *idxs: hcl_math.max(-beta.v - 0.1, vf[idxs]))

        # Initialize the Hamiltonian tensor
        h = hcl.compute(self.grid.shape, lambda *idxs: vf[idxs], name='h')
        vf_inter = hcl.compute(self.grid.shape, lambda *idxs: vf[idxs], name='vf_inter')


        """ UPDATE FOR THE STATES IN I """
        # Compute Hamiltonian term, max and min derivative
        self.hamiltonian_stage(h, beta, vf, t, xs, p, 0,
                               dv_min=dv_min, dv_max=dv_max, dv_diff=dv_diff)
        
        if self.debug:
            hcl.update(h_dbg, lambda *idxs: h[idxs])

        # Compute artificial dissipation
        self.dissipation_stage(h, beta, vf, t, xs, p, 0,
                               dv_min=dv_min, dv_max=dv_max, dv_diff=dv_diff,
                               max_alpha=max_alpha)
        


        # Compute integration time step
        delta_t = hcl.compute((1,), lambda _: self.time_step(t, 0, max_alpha=max_alpha))


        # First order Runge-Kutta (RK) integrator to update the value function in the immediate update set
        # -1 update the intermediate value function 
        # 0 use the vf_inter to calculate P
        # 1 compute vf for both P and I 
    
        self.value_function_update(h, beta, vf_inter, delta_t, p, sum_exp, num_exp, -1)
        self.value_function_update(h, beta, vf_inter, delta_t, p, sum_exp, num_exp, 0)
        #used to be instead:
        #hcl.update(vf, lambda *idxs: vf[idxs] + h[idxs] * delta_t.v)

        
        
        """ UPDATE P TO FIND ITS FINAL VERSION """
#this should be removed and DONE IN STEP 0
        # The problem is likely here, since the boarder of p has to propagate faster, considering more neighbours. 
        # By now it only can move by 1 in any direction
 #       self.potential_set_check(beta, vf, p)
 # COMMMENTED FOR NOW!!!!

        """ UPDATE FOR THE STATES IN P """
        # Compute Hamiltonian term, max and min derivative for states in P
        self.hamiltonian_stage(h, beta, vf, t, xs, p, 1,
                               dv_min=dv_min, dv_max=dv_max, dv_diff=dv_diff)
        

        #hcl.update(max_alpha, lambda _: -1e9)


        # Dissipation including P
        self.dissipation_stage(h, beta, vf, t, xs, p, 1,
                               dv_min=dv_min, dv_max=dv_max, dv_diff=dv_diff,
                               max_alpha=max_alpha)
        

        delta_t_final = hcl.compute((1,), lambda _: self.time_step(t, 1, max_alpha=max_alpha))


        # Final value function update
        self.value_function_update(h, beta, vf, delta_t_final, p, sum_exp, num_exp, 1)

        
        





    def value_function_update(self, h, beta, vf, delta_t, p, sum_exp, num_exp, type):
        """" Update the value function for all cells in immediate update set """

        def body(*idxs):
            if type <= 0:
                cond = hcl_math.abs(vf[idxs]) < beta.v 
            else:
                cond = hcl.or_((p[idxs] == 1),  (hcl_math.abs(vf[idxs]) < beta.v))
                
            with hcl.if_(cond):
                if (type != 0):
                    vf[idxs] = vf[idxs] + h[idxs] * delta_t.v 
                # CALCULATE THE NEW P HERE
                else:
                    with hcl.if_(hcl_math.abs(vf[idxs]) < beta.v):
                        for axis in range(self.grid.ndims):
                            axis_len = vf.shape[axis]

                            # left and right element
                            lx = list(idxs)
                            rx = list(idxs)
                            lx[axis] -= 1
                            rx[axis] += 1
                            if self.grid.periodic_dims[axis]:
                                with hcl.if_(lx[axis] == -1):
                                    lx[axis] = axis_len - 1
                                with hcl.if_(rx[axis] == axis_len):
                                    rx[axis] = 0
                            lx = tuple(lx)
                            rx = tuple(rx)

                            with hcl.if_(hcl.and_(lx[axis] != -1, rx[axis] != axis_len)):

                                # calculation of P for the left element
                                with hcl.if_(hcl.and_(hcl_math.abs(vf[lx]) >= beta.v, hcl_math.abs(vf[rx]) < beta.v)):  
                                    dif_vf = hcl.scalar(0, 'dif_vf')
                                    dif_vf.v = vf[idxs] - vf[rx]
                                    num_exp[lx] += 1
                                    sum_exp[lx] += vf[idxs] + dif_vf.v
                                    exp_vf = hcl.scalar(sum_exp[lx] / num_exp[lx], 'exp_vf')
                                    with hcl.if_(hcl_math.abs(exp_vf.v) < beta.v):
                                        p[lx] = 1
                                    with hcl.else_():
                                        p[lx] = 0

                                # calculation of P for the right element
                                with hcl.if_(hcl.and_(hcl_math.abs(vf[rx]) >= beta.v, hcl_math.abs(vf[lx]) < beta.v)):  
                                    dif_vf = hcl.scalar(0, 'dif_vf')
                                    dif_vf.v = vf[idxs] - vf[lx]
                                    num_exp[rx] += 1
                                    sum_exp[rx] += vf[idxs] + dif_vf.v
                                    exp_vf = hcl.scalar(sum_exp[rx] / num_exp[rx], 'exp_vf')
                                    with hcl.if_(hcl_math.abs(exp_vf.v) < beta.v):
                                        p[rx] = 1
                                    with hcl.else_():
                                        p[rx] = 0


                                

        hcl.mutate(vf.shape, body, name="Value_Function")

    def potential_set_check(self, beta, vf, p):

        """ Check if the states in the current P version should be updated """
        def body(*idxs):
            with hcl.if_(p[idxs] == 1):
                sum_exp = hcl.scalar(0, 'sum_exp')
                num = hcl.scalar(0, 'num')
                for axis in range(self.grid.ndims):
                    axis_len = vf.shape[axis]

                    # check based on the left element
                    in1 = list(idxs)
                    in1[axis] -= 1
                    in2 = list(idxs)
                    in2[axis] -= 2

                    

                    if self.grid.periodic_dims[axis]:
                        with hcl.if_(in1[axis] == -1):
                            in1[axis] = axis_len - 1
                            in2[axis] = axis_len - 2
                        with hcl.if_(in2[axis] == -1):
                            in2[axis] = axis_len - 1

                    in1 = tuple(in1)
                    in2 = tuple(in2)
                    with hcl.if_(hcl.and_(
                        in1[axis] >= 0,
                        in2[axis] >= 0,
                        hcl_math.abs(vf[in1]) < beta.v,
                        hcl_math.abs(vf[in2]) < beta.v
                    )):
                        dif_vf = hcl.scalar(0, 'dif_vf')
                        dif_vf.v = vf[in1] - vf[in2]
                        sum_exp.v += vf[in1] + dif_vf.v
                        num.v += 1
                
                        
                    # check based on the right element
                    in1 = list(idxs)
                    in1[axis] += 1
                    in2 = list(idxs)
                    in2[axis] += 2
                    if self.grid.periodic_dims[axis]:
                        with hcl.if_(in1[axis] == axis_len):
                            in1[axis] = 0
                            in2[axis] = 1
                        with hcl.if_(in2[axis] == axis_len):
                            in2[axis] = 0
                    
                    in1 = tuple(in1)
                    in2 = tuple(in2)
                    with hcl.if_(hcl.and_(
                        in1[axis] < axis_len,
                        in2[axis] < axis_len,
                        hcl_math.abs(vf[in1]) < beta.v,
                        hcl_math.abs(vf[in2]) < beta.v
                    )):
                        dif_vf = hcl.scalar(0, 'dif_vf')
                        dif_vf.v = vf[in1] - vf[in2]
                        sum_exp.v += vf[in1] + dif_vf.v
                        num.v += 1

                # check if this state should be kept in P 
                with hcl.if_(num.v != 0):
                    exp_vf = hcl.scalar(sum_exp.v / num.v, 'exp_vf')
                    with hcl.if_(hcl_math.abs(exp_vf.v) >= beta.v):
                        p[idxs] = 0
                with hcl.else_():
                    p[idxs] = 0
                


        hcl.mutate(p.shape, body, name="Potential_Update")
        
        


    def hamiltonian_stage(self, h, beta, vf, t, xs, p, type, *,
                          dv_min, dv_max, dv_diff):
        """Calculate Hamiltonian for every grid point in V_init"""

        def body(*idxs):
           # cond = hcl.or_(hcl_math.abs(vf[idxs]) < beta.v, p[idxs] == 1)
            if type == 0:
                cond = hcl_math.abs(vf[idxs]) < beta.v 
            else:
                cond = p[idxs] == 1

            with hcl.if_(cond):

                u = hcl.compute(self.ctrl_shape, lambda _: 0, name='u')
                d = hcl.compute(self.dstb_shape, lambda _: 0, name='d')
                x = hcl.compute(self.state_shape, lambda _: 0, name='x')
                dx = hcl.compute(self.state_shape, lambda _: 0, name='dx')
                dv = hcl.compute(self.state_shape, lambda _: 0, name='dv_avg')

                # x_n = X_{n,i} where 
                #   x = `x` The state tensor,
                #   X = `xs` The list of state space arrays,
                #   n = Current state dimension (in updating x),
                #   i = `idxs[n]` Index of current grid point in dimension `n`,
                for n, i in enumerate(idxs):
                    x[n] = xs[n][i]

                for axis in range(self.grid.ndims):

                    left = hcl.scalar(0, 'left')
                    right = hcl.scalar(0, 'right')

                    # Compute the spatial derivative of the value function (dV/dx)
                    spatial_derivative(left, right, axis, vf, self.grid, *idxs)

                    # do for both left/right derivatives
                    for deriv in (left, right):
                        
                        with hcl.if_(deriv.v < dv_min[axis]):
                            dv_min[axis] = deriv.v
                        with hcl.if_(dv_max[axis] < deriv.v):
                            dv_max[axis] = deriv.v
                    
                    # checking if the values outside of I were considered in the derivatives
                   
                    dv[axis] = (left.v + right.v) / 2
                    
                    dv_diff[idxs + (axis,)] = right.v - left.v

                # Use the model's methods to solve optimal control
                self.model.opt_ctrl(u, dv, t, x)
                self.model.opt_dstb(d, dv, t, x)

                # Calculate dynamical rates of changes
                self.model.dynamics(dx, t, x, u, d)

                # Calculate Hamiltonian terms
                h[idxs] = hcl_math.dot(dv, dx)

        hcl.mutate(vf.shape, body, name='Hamiltonian')

    def dissipation_stage(self, h, beta, vf, t, xs, p, type, *,
                            dv_min, dv_max, dv_diff, 
                            max_alpha):
        """Calculate the dissipation"""

        def body(*idxs):
            
            #cond = hcl.or_(hcl_math.abs(vf[idxs]) < beta.v, p[idxs] == 1)
            if type == 0:
                cond = hcl_math.abs(vf[idxs]) < beta.v 
            else:
                cond = p[idxs] == 1

            with hcl.if_(cond):
                x = hcl.compute(self.state_shape, lambda _: 0, name='x')

                # Each has a combination of lower/upper bound on control and disturbance
                dx_ll = hcl.compute(self.state_shape, lambda _: 0, name='dx_ll')
                dx_lu = hcl.compute(self.state_shape, lambda _: 0, name='dx_lu')
                dx_ul = hcl.compute(self.state_shape, lambda _: 0, name='dx_ul')
                dx_uu = hcl.compute(self.state_shape, lambda _: 0, name='dx_uu')

                lower_opt_ctrl = hcl.compute(self.ctrl_shape, lambda _: 0, name='lower_opt_ctrl')
                upper_opt_ctrl = hcl.compute(self.ctrl_shape, lambda _: 0, name='upper_opt_ctrl')
                lower_opt_dstb = hcl.compute(self.dstb_shape, lambda _: 0, name='lower_opt_dstb')
                upper_opt_dstb = hcl.compute(self.dstb_shape, lambda _: 0, name='upper_opt_dstb')

                for n, i in enumerate(idxs):
                    x[n] = xs[n][i]

                # Find LOWER BOUND optimal disturbance
                self.model.opt_dstb(lower_opt_dstb, dv_min, t, x)

                # Find UPPER BOUND optimal disturbance
                self.model.opt_dstb(upper_opt_dstb, dv_max, t, x)

                # Find LOWER BOUND optimal control
                self.model.opt_ctrl(lower_opt_ctrl, dv_min, t, x)

                # Find UPPER BOUND optimal control
                self.model.opt_ctrl(upper_opt_ctrl, dv_max, t, x)

                # Find magnitude of rates of changes
                self.model.dynamics(dx_ll, t, x, lower_opt_ctrl, lower_opt_dstb)
                hcl.update(dx_ll, lambda i: hcl_math.abs(dx_ll[i]))

                self.model.dynamics(dx_lu, t, x, lower_opt_ctrl, upper_opt_dstb)
                hcl.update(dx_lu, lambda i: hcl_math.abs(dx_lu[i]))

                self.model.dynamics(dx_ul, t, x, upper_opt_ctrl, lower_opt_dstb)
                hcl.update(dx_ul, lambda i: hcl_math.abs(dx_ul[i]))

                self.model.dynamics(dx_uu, t, x, upper_opt_ctrl, upper_opt_dstb)
                hcl.update(dx_uu, lambda i: hcl_math.abs(dx_uu[i]))

                # Calulate alpha
                alpha = hcl.compute(self.state_shape, 
                                        lambda i: hcl_math.max(dx_ll[i], dx_lu[i], 
                                                            dx_ul[i], dx_uu[i]), 
                                        name='alpha')

                hcl.update(max_alpha, lambda i: hcl_math.max(max_alpha[i], alpha[i]))

                # Finally we update the hamiltonian
                # dv_diff has shape <grid...> x <states>. Here we use dv_diff at the current grid point.
                dv_diff_here = hcl.compute(self.state_shape, lambda n: dv_diff[idxs + (n,)])
                dissipation_v = hcl_math.dot(dv_diff_here, alpha) / 2
                h[idxs] = (h[idxs] + dissipation_v)

        hcl.mutate(h.shape, body, name='Dissipation')

    def time_step(self, t, type, *, 
                  max_alpha):

        step_bound = hcl.scalar(0, 'step_bound')

        tmp = hcl.scalar(0)
        for i, res in enumerate(self.grid.dx):
            tmp.v += max_alpha[i] / res
        step_bound.v = 0.8 / tmp.v

        if (type == 1):
            with hcl.if_(t[1] - t[0] < step_bound.v):
                step_bound.v = t[1] - t[0]

            t[0] += step_bound.v

        return step_bound.v


class HJSolverSP(Solver):

    def __init__(self, grid, tau, model, beta=1, **kwargs):
        
        self.tau = np.asarray(tau)

        self.beta = beta

        super().__init__(grid, model, **kwargs)

    def __call__(self, *,
                 target, target_mode='min',
                 constraint=None, constraint_mode='max'):
        
        target_invariant = target.shape == self.grid.shape
        assert target_invariant or target.shape == self.grid.shape + self.tau.shape
        assert target_mode in ('max', 'min')

        if constraint is not None:
            constraint_invariant = constraint.shape == self.grid.shape
            assert constraint_invariant or constraint.shape == self.grid.shape + self.tau.shape
            assert constraint_mode in ('max', 'min')

        # Tensor input to our computation graph
        vf = target if target_invariant else target[..., -1]
        t = np.flip(self.tau)
        xs = [ax.flatten() for ax in self.grid.vs]

        # Extend over time axis
        out = np.zeros(vf.shape + (len(t),))
        out[..., -1] = vf

        ################ USE THE EXECUTABLE ############
        # Variables used for timing
        execution_time = 0
        now = t[-1]

        if self.interactive:
            print("Running...")
            line_length = 0

        # Backward reachable set/tube will be computed over the specified time horizon
        # Or until convergent ( which ever happens first )
        for i in reversed(range(0, len(t)-1)):

            vf = out[..., i+1].copy()

            pde_args = [np.array([self.beta]), vf, np.array([now, t[i]]), *xs]
            pde_args = list(map(hcl.asarray, pde_args))

            if self.debug:
                h = hcl.asarray(np.zeros_like(vf))
                pde_args = [h] + pde_args

            while now <= t[i] - 1e-4:

                # Start timing
                start = time.time()

                # Run the execution and pass input into graph
                self._executable(*pde_args)

                # End timing
                end = time.time()

                # Calculate computation time
                execution_time += end - start

                # Get current time from within solver
                if self.debug:
                    now = pde_args[3].asnumpy()[0]
                else:
                    now = pde_args[2].asnumpy()[0]

                # Some information printing
                if self.interactive:
                    line = f"> [{execution_time:.2f}: {i}] Integration time: {end - start:.5f} s - Now: {now:.5f}"
                    line_length = max(line_length, len(line))
                    print(line, end="\r", flush=True)

            if self.debug:
                h = pde_args[1].asnumpy()
                vf = pde_args[2].asnumpy()
            else:
                vf = pde_args[1].asnumpy()

            op = np.minimum if target_mode == 'min' else np.maximum
            vf = op(vf, target) if target_invariant else op(vf, target[..., i])

            if constraint is not None:
                op = np.minimum if constraint_mode == 'min' else np.maximum
                vf = op(vf, -constraint) if constraint_invariant else op(vf, -constraint[..., i])

            out[..., i] = vf

        # Time info printing
        if self.interactive:
            line = f"> Total kernel time: {execution_time:.2f} s"
            line = line.ljust(line_length)
            print(line, end="\n\n")

        # Flip time axis so that earliest time is first
        # out = np.flip(out, axis=-1)

        return out