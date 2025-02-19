from math import pi

from pyspect import *
from pyspect.langs.ltl import *

TLT.select(ContinuousLTL)

## CONSTANTS ##

LONVEL = (0.3, 0.6) # [m/s]
LATVEL = (0, 0.2)   # [m/s]

RBOUND = (0.8, 1.6) # [m]

RI = 1.0    # [m]
RM = 1.3    # [m]
RO = 1.6    # [m]

PHI_LOOKAHEAD = 2 # [rad]
TIME_HORIZON = 2 # [s]; comp time horizon

## TRAFFIC RULES ##

lonspeed = BoundedSet(v_phi=LONVEL)
latspeed = BoundedSet(v_r=LATVEL)

## ROUNDABOUT ##

lanes = And(BoundedSet(r=(RI, RO)), lonspeed, latspeed)

inner = And(BoundedSet(r=(RI, RM)), lanes)
outer = And(BoundedSet(r=(RM, RO)), lanes)

## TASKS ##

stay_inner = Until(inner, inner)
stay_outer = Until(outer, outer)
goto_inner = Until(lanes, inner)
goto_outer = Until(lanes, outer)

import numpy as np
import hj_reachability as hj

class HJImpl:

    solver_settings = hj.SolverSettings.with_accuracy("low")

    def __init__(self, dynamics, grid, time_horizon):
        self.grid = grid
        self.dynamics = dynamics
        self.timeline = self.new_timeline(time_horizon)
           
    def new_timeline(self, target_time, start_time=0, time_step=0.2):
        assert time_step > 0
        is_forward = target_time >= start_time
        target_time += 1e-5 if is_forward else -1e-5
        time_step *= 1 if is_forward else -1
        return np.arange(start_time, target_time, time_step)


    def set_axes_names(self, time: str, *names: str) -> None:
        assert len(names) == self.grid.ndim
        # HJImpl uses time-state dims convention
        self._axes_names = (time, *names)
        self.ndim = len(self._axes_names) 

    def assert_axis(self, ax: int | str) -> None:
        match ax:
            case int(i):
                assert -len(self._axes_names) <= i < len(self._axes_names), \
                    f'Axis ({i=}) does not exist.'
            case str(name):
                assert name in self._axes_names, \
                    f'Axis ({name=}) does not exist.'

    def axis(self, ax: int | str) -> int:
        self.assert_axis(ax)
        match ax:
            case int(i):
                return i
            case str(name):
                return self._axes_names.index(name)

    def axis_name(self, i: int) -> str:
        self.assert_axis(i)
        return self._axes_names[i]

    def axis_is_periodic(self, ax: int | str) -> bool:
        i = self.axis(ax)
        # HJImpl uses time-state convention
        return bool(self.grid._is_periodic_dim[i-1])


    def axis_vec(self, ax: int | str):
        i = self.axis(ax)
        # HJImpl uses time-state convention
        if i == 0:
            shape = (*self.timeline.shape, *[1]*self.grid.ndim)
            return self.timeline.reshape(shape)
        else:
            shape = (1, *self.grid.shape)
            return self.grid.states[..., i-1].reshape(shape)
        

    def plane_cut(self, normal, offset, axes=None):
        # HJImpl uses time-state convention
        shape = (len(self.timeline), *self.grid.shape)
        data = np.zeros(shape)
        axes = axes or list(range(self.ndim))
        for i, k, m in zip(axes, normal, offset):
            data -= k*self.axis_vec(i) - k*m
        return data


    def empty(self):
        shape = (len(self.timeline), *self.grid.shape)
        return np.ones(shape)*np.inf
    
    def complement(self, vf):
        return np.asarray(-vf)
    
    def intersect(self, vf1, vf2):
        return np.maximum(vf1, vf2)

    def union(self, vf1, vf2):
        return np.minimum(vf1, vf2)
    
    def reachF(self, target, constraints=None):
        self.dynamics.with_mode('reach')
        vf = hj.solve(self.solver_settings,
                      self.dynamics,
                      self.grid,
                      self.timeline,
                      target,
                      constraints)
        return np.asarray(vf)

    def reach(self, target, constraints=None):
        self.dynamics.with_mode('reach')
        if not self.is_invariant(target):
            target = np.flip(target, axis=0)
        if not self.is_invariant(constraints):
            constraints = np.flip(constraints, axis=0)
        vf = hj.solve(self.solver_settings,
                      self.dynamics,
                      self.grid,
                      -self.timeline,
                      target,
                      constraints)
        return np.flip(np.asarray(vf), axis=0)
    
    def avoid(self, target, constraints=None):
        self.dynamics.with_mode('avoid')
        if not self.is_invariant(target):
            target = np.flip(target, axis=0)
        if not self.is_invariant(constraints):
            constraints = np.flip(constraints, axis=0)
        vf = hj.solve(self.solver_settings,
                      self.dynamics,
                      self.grid,
                      -self.timeline,
                      target,
                      constraints)
        return np.flip(np.asarray(vf), axis=0)


    def project_onto(self, vf, *idxs, keepdims=False, union=True):
        idxs = [len(vf.shape) + i if i < 0 else i for i in idxs]
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

class HJImplDebugShape(HJImpl):
    
    @staticmethod
    def _shape(vf):
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
        print(s := f'PLANE{self._shape(data)}' + '\n')
        return data, s

    def empty(self):
        out = super().empty()
        print(s := f'EMPTY{self._shape(out)}' + '\n')
        return out, s
        
    def complement(self, inp):
        print(head := 'neg(')
        
        vf, s = inp
        print(s := self._indent(s) + ',')
        
        out = super().complement(vf)
        print(tail := f') => {self._shape(out)}', '\n')
        return out, '\n'.join([head, s, tail])
        
    
    def intersect(self, inp1, inp2):
        print(head := 'max(')

        vf1, s1 = inp1
        vf2, s2 = inp2

        print(s1 := self._indent(s1) + ',')
        print(s2 := self._indent(s2) + ',')

        out = super().intersect(vf1, vf2)
        print(tail := f') => {self._shape(out)}', '\n')
        return out, '\n'.join([head, s1, s2, tail])

    def union(self, inp1, inp2):
        print(head := 'min(')

        vf1, s1 = inp1
        vf2, s2 = inp2

        print(s1 := self._indent(s1) + ',')
        print(s2 := self._indent(s2) + ',')

        out = super().union(vf1, vf2)
        print(tail := f') => {self._shape(out)}', '\n')
        return out, '\n'.join([head, s1, s2, tail])
    
    def reach(self, target, constraints=None):
        print(head := 'reach(')

        vf1, s1 = target
        vf2, s2 = constraints

        print(s1 := self._indent(s1) + ',')
        print(s2 := self._indent(s2) + ',')

        out = super().reach(vf1, vf2)
        print(tail := f') => {self._shape(out)}', '\n')
        return out, '\n'.join([head, s1, s2, tail])

from hj_reachability.systems import Bicycle5DCircular
from pyspect.plotting.levelsets import *

reach_dynamics = Bicycle5DCircular(min_steer=-pi/4,
                                   max_steer=+pi/4,
                                   min_accel=-0.4,
                                   max_accel=+0.4)

#                        r,                  phi,  v_r, v_phi,   v_yaw
min_bounds = np.array([0.8, 0.5 * -PHI_LOOKAHEAD, -0.2,  +0.3,   -pi/2])
max_bounds = np.array([1.8,       +PHI_LOOKAHEAD, +0.2,  +0.7,   +pi/2])
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(min_bounds, max_bounds),
                                                               (11, 51, 5, 5, 5),
                                                               periodic_dims=1)

impl = HJImplDebugShape(reach_dynamics, grid, TIME_HORIZON)
impl.set_axes_names('t', 'r', 'phi', 'v_r', 'v_phi', 'v_yaw')

# out, s = TLT(goto_outer).realize(impl)
# print(s)

win_arr = np.ones((len(impl.timeline), *grid.shape))
win_arr[3:9] = -1
win = Set((win_arr, f'WIN<{"x".join(map(str, win_arr.shape))}>'))

out, s = And(BoundedSet(r=(-0.5, +0.5), phi=(-0.5, +0.5)), win).realize(impl)
# out = goto_outer.realize(impl)

plot3D_levelset(impl.project_onto(out, 0, 1, 2),
                min_bounds=[0, *min_bounds[:2]],
                max_bounds=[3, *max_bounds[:2]],
                xtitle="r [m]", ytitle="phi [rad]")