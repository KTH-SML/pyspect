import numpy as np
import hj_reachability as hj
from .axes import AxesImpl

# Hamilton-Jacobi Reachability
# ----------------------------

class HJImpl(AxesImpl):

    TIME_STEP = 0.02
    SOLVER_SETTINGS = hj.SolverSettings.with_accuracy("low")

    def __init__(self, dynamics, axis_names, min_bounds, max_bounds, grid_shape, time_horizon, time_step=...):
        super().__init__(axis_names, min_bounds, max_bounds)

        if time_step is Ellipsis: time_step = self.TIME_STEP
        self.timeline = self.new_timeline(time_horizon, time_step=time_step)

        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(min_bounds, max_bounds),
            grid_shape,
            periodic_dims=tuple(i-1 for i in range(self.ndim) if self.axis_is_periodic(i)),
        )

        self._dt = time_step
        self._dx = self.grid.spacings

        self.shape = self.grid.shape

        Dynamics = dynamics.pop('cls')
        if Dynamics is not None:
            self.reach_dynamics = Dynamics(**dynamics).with_mode('reach')
            self.avoid_dynamics = Dynamics(**dynamics).with_mode('avoid')

    ## Auxiliary Methods ##

    def new_timeline(self, target_time, start_time=0, time_step=...):
        if time_step is Ellipsis: time_step = self.TIME_STEP
        assert time_step > 0
        is_forward = target_time >= start_time
        target_time += 1e-5 if is_forward else -1e-5
        time_step *= 1 if is_forward else -1
        return np.arange(start_time, target_time, time_step)

    def axis_step(self, ax: int | str):
        i = self.axis(ax)
        return self._dx[i]

    def axis_vec(self, ax: int | str):
        i = self.axis(ax)
        shape = tuple(n if j == i else 1
                      for j, n in enumerate(self.shape))
        return self.grid.coordinate_vectors[i].reshape(shape)

    def project_onto(self, vf, *axes, keepdims=False, union=True):
        axes = [self.axis(ax) for ax in axes]
        idxs = [len(vf.shape) + i if i < 0 else i for i in axes]
        dims = [i for i in range(len(vf.shape)) if i not in idxs]
        if union:
            return vf.min(axis=tuple(dims), keepdims=keepdims)
        else:
            return vf.max(axis=tuple(dims), keepdims=keepdims)

    ## pyspect Interfaces ##        

    def plane_cut(self, normal, offset, axes=None):
        axes = axes or list(range(self.ndim))
        axes = [self.axis(i) for i in axes]

        assert len(axes) == len(normal) == len(offset)

        data = np.zeros(self.shape)
        for i, k, m in zip(axes, normal, offset):
            # Doesn't contribute
            if k == 0: continue
            
            # NOTE: Without this condition, problems may arise that are VERY ANNOYING to debug...
            amin, amax = self.axis_bounds(i)
            assert amin < m < amax, (
                f'For axis {i} ({self.axis_name(i)}): '
                f'Offset ({m}) must be inside boundary ({(amin, amax)}) to avoid numerical instability'
            )

            # normalize wrt discretization 
            k /= self.axis_step(i)
            data -= k*self.axis_vec(i) - k*m
        
        return data

    def empty(self):
        return np.ones(self.shape)*np.inf
    
    def complement(self, vf):
        return np.asarray(-vf)
    
    def intersect(self, vf1, vf2):
        return np.maximum(vf1, vf2)

    def union(self, vf1, vf2):
        return np.minimum(vf1, vf2)
    
    def reachF(self, target, constraints=None):
        vf = hj.solve(self.SOLVER_SETTINGS,
                      self.avoid_dynamics,
                      self.grid,
                      self.timeline,
                      target,
                      constraints,
                      stack=False)
        return np.asarray(vf)

    def reach(self, target, constraints=None):
        vf = hj.solve(self.SOLVER_SETTINGS,
                      self.reach_dynamics,
                      self.grid,
                      -self.timeline,
                      target,
                      constraints,
                      stack=False)
        return np.asarray(vf)
    
    def avoid(self, target, constraints=None):
        vf = hj.solve(self.SOLVER_SETTINGS,
                      self.avoid_dynamics,
                      self.grid,
                      -self.timeline,
                      target,
                      constraints,
                      stack=False)
        return np.asarray(vf)

    def rci(self, target):
        return self.complement(self.avoid(self.complement(target)))


# Time-Varying Hamilton-Jacobi Reachability
# -----------------------------------------

class TVHJImpl(AxesImpl):

    TIME_STEP = 0.02
    SOLVER_SETTINGS = hj.SolverSettings.with_accuracy("low")

    def __init__(self, dynamics, axis_names, min_bounds, max_bounds, grid_shape, time_horizon, time_step=...):
        super().__init__((         't', *axis_names), 
                         [           0, *min_bounds],
                         [time_horizon, *max_bounds])

        if time_step is Ellipsis: time_step = self.TIME_STEP
        self.timeline = self.new_timeline(time_horizon, time_step=time_step)

        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(min_bounds, max_bounds),
            grid_shape,
            periodic_dims=tuple(i-1 for i in range(self.ndim) if self.axis_is_periodic(i)),
        )

        self._dt = time_step
        self._dx = self.grid.spacings

        self.shape = (len(self.timeline), *self.grid.shape)
        self._shape_inv = (1, *self.grid.shape) # Used for constants

        Dynamics = dynamics.pop('cls')
        if Dynamics is not None:
            self.reach_dynamics = Dynamics(**dynamics).with_mode('reach')
            self.avoid_dynamics = Dynamics(**dynamics).with_mode('avoid')
           
    ## Auxiliary Methods ##

    def new_timeline(self, target_time, start_time=0, time_step=...):
        if time_step is Ellipsis: time_step = self.TIME_STEP
        assert time_step > 0
        is_forward = target_time >= start_time
        target_time += 1e-5 if is_forward else -1e-5
        time_step *= 1 if is_forward else -1
        return np.arange(start_time, target_time, time_step)

    # NOTE: TVHJImpl uses time-state convention

    def axis_step(self, ax: int | str):
        i = self.axis(ax)
        return self._dt if i == 0 else self._dx[i-1]

    def axis_vec(self, ax: int | str):
        i = self.axis(ax)
        shape = tuple(n if j == i else 1
                      for j, n in enumerate(self.shape))
        # TVHJImpl uses time-state convention
        return (self.timeline.reshape(shape) if i == 0 else
                self.grid.coordinate_vectors[i-1].reshape(shape))

    def project_onto(self, vf, *axes, keepdims=False, union=True):
        axes = [self.axis(ax) for ax in axes]
        idxs = [len(vf.shape) + i if i < 0 else i for i in axes]
        dims = [i for i in range(len(vf.shape)) if i not in idxs]
        if union:
            return vf.min(axis=tuple(dims), keepdims=keepdims)
        else:
            return vf.max(axis=tuple(dims), keepdims=keepdims)

    def is_invariant(self, vf):
        return (True if vf is None else
                vf.shape[0] == 1 if len(vf.shape) == self.ndim else
                len(vf.shape) == len(self.grid.shape))

    def make_tube(self, vf):
        return (vf if not self.is_invariant(vf) else
                np.concatenate([vf[np.newaxis, ...]] * len(self.timeline)))

    ## pyspect Interfaces ##        

    def plane_cut(self, normal, offset, axes=None):
        axes = axes or list(range(self.ndim))
        axes = [self.axis(i) for i in axes]

        assert len(axes) == len(normal) == len(offset)
        
        data = np.zeros(self.shape if 0 in axes else self._shape_inv)
        for i, k, m in zip(axes, normal, offset):
            # Doesn't contribute
            if k == 0: continue
            
            # NOTE: Without this condition, problems may arise that are VERY ANNOYING to debug...
            amin, amax = self.axis_bounds(i)
            assert amin < m < amax, (
                f'For axis {i} ({self.axis_name(i)}): '
                f'Offset ({m}) must be inside boundary ({(amin, amax)}) to avoid numerical instability'
            )

            # normalize wrt discretization 
            k /= self.axis_step(i)
            data -= k*self.axis_vec(i) - k*m
        
        return data

    def empty(self):
        return np.ones(self._shape_inv)*np.inf
    
    def complement(self, vf):
        return np.asarray(-vf)
    
    def intersect(self, vf1, vf2):
        return np.maximum(vf1, vf2)

    def union(self, vf1, vf2):
        return np.minimum(vf1, vf2)
    
    def reachF(self, target, constraints=None):
        vf = hj.solve(self.SOLVER_SETTINGS,
                      self.avoid_dynamics,
                      self.grid,
                      self.timeline,
                      target,
                      constraints)
        return np.asarray(vf)

    def reach(self, target, constraints=None):
        if not self.is_invariant(target):
            target = np.flip(target, axis=0)
        if not self.is_invariant(constraints):
            constraints = np.flip(constraints, axis=0)
        vf = hj.solve(self.SOLVER_SETTINGS,
                      self.reach_dynamics,
                      self.grid,
                      -self.timeline,
                      target,
                      constraints)
        return np.flip(np.asarray(vf), axis=0)
    
    def avoid(self, target, constraints=None):
        if not self.is_invariant(target):
            target = np.flip(target, axis=0)
        if not self.is_invariant(constraints):
            constraints = np.flip(constraints, axis=0)
        vf = hj.solve(self.SOLVER_SETTINGS,
                      self.avoid_dynamics,
                      self.grid,
                      -self.timeline,
                      target,
                      constraints)
        return np.flip(np.asarray(vf), axis=0)

    def rci(self, target):
        return self.complement(self.avoid(self.complement(target)))


class TVHJImplDebugShape(TVHJImpl):
    
    @staticmethod
    def _shape_str(vf):
        return f'<{"x".join(map(str, vf.shape))}>'
    
    @staticmethod
    def _indent(s: str, n: int = 2, skip_first=False) -> str:
        out = []
        lines = s.splitlines()
        if skip_first:
            line, *lines = lines
            out.append(line)
        out += [' '*n + line for line in lines]
        return '\n'.join(out)

    def plane_cut(self, *args, **kwds):
        data = super().plane_cut(*args, **kwds)
        print(s := f'PLANE{self._shape_str(data)}' + '\n')
        return data, s

    def empty(self):
        out = super().empty()
        print(s := f'EMPTY{self._shape_str(out)}' + '\n')
        return out, s
        
    def complement(self, inp):
        print(head := 'neg(')
        
        vf, s = inp
        print(s := self._indent(s) + ',')
        
        out = super().complement(vf)
        print(tail := f') => {self._shape_str(out)}', '\n')
        return out, '\n'.join([head, s, tail])
        
    
    def intersect(self, inp1, inp2):
        print(head := 'max(')

        vf1, s1 = inp1
        vf2, s2 = inp2

        print(s1 := self._indent(s1) + ',')
        print(s2 := self._indent(s2) + ',')

        out = super().intersect(vf1, vf2)
        print(tail := f') => {self._shape_str(out)}', '\n')
        return out, '\n'.join([head, s1, s2, tail])

    def union(self, inp1, inp2):
        print(head := 'min(')

        vf1, s1 = inp1
        vf2, s2 = inp2

        print(s1 := self._indent(s1) + ',')
        print(s2 := self._indent(s2) + ',')

        out = super().union(vf1, vf2)
        print(tail := f') => {self._shape_str(out)}', '\n')
        return out, '\n'.join([head, s1, s2, tail])
    
    def reach(self, target, constraints=None):
        print(head := 'reach(')

        vf1, s1 = target
        vf2, s2 = constraints

        print(s1 := self._indent(s1) + ',')
        print(s2 := self._indent(s2) + ',')

        out = super().reach(vf1, vf2)
        print(tail := f') => {self._shape_str(out)}', '\n')
        return out, '\n'.join([head, s1, s2, tail])