"""
Hamilton-Jacobi reachability implementations using the hj_reachability package.

Requires:
    - hj_reachability
    - jax_tqdm
"""

import functools
from pydoc import text

import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import hj_reachability as hj

from .axes import *
from .plotly import *

__all__ = [
    'TVHJImpl',
    'TVHJImplDebugger',
]

type LevelSet = jnp.ndarray


# Time-Varying Hamilton-Jacobi Reachability
# -----------------------------------------

class TVHJImpl(PlotlyImpl[LevelSet], AxesImpl[LevelSet]):

    SOLVER_SETTINGS = hj.SolverSettings.with_accuracy("low")

    def __init__(self, dynamics, axis_specs, _stack=True):
        
        # Add time axis
        super().__init__(axis_specs)
        assert axis_specs[0]['name'] == 't', "First axis must be time ('t')"

        min_bounds = [spec['bounds'][0] for spec in axis_specs]
        max_bounds = [spec['bounds'][1] for spec in axis_specs]
        grid_shape = [(spec['points'] if 'points' in spec else
                       int(abs(max_bounds[i] - min_bounds[i]) // spec['step']) + 1)
                      for i, spec in enumerate(axis_specs)]

        self.timeline = jnp.linspace(min_bounds[0], max_bounds[0], grid_shape[0])

        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(min_bounds[1:], max_bounds[1:]),
            grid_shape[1:],
            periodic_dims=tuple(i-1 for i in range(self.ndim) if self.axis_is_periodic(i)),
        )

        self._dt = self.timeline[1] - self.timeline[0]
        self._dx = self.grid.spacings

        self.shape = (len(self.timeline), *self.grid.shape)
        self._shape_inv = (1, *self.grid.shape) # Used for constants

        self._stack = _stack # EXPERIMENTAL: Whether to stack time dimension in outputs. Don't turn off unless you know what you're doing.

        Dynamics = dynamics.pop('cls')
        if Dynamics is not None:
            self.reach_dynamics = Dynamics(**dynamics).with_mode('reach')
            self.avoid_dynamics = Dynamics(**dynamics).with_mode('avoid')
   
    
    @functools.partial(jax.jit, static_argnames=("self", "dynamics", "progress_bar", "stack"))
    def solve(self, solver_settings, dynamics, grid, times, target, constraint=None, progress_bar=True, stack=True):
            
        is_target_invariant = self.is_invariant(target)
        is_constraint_invariant = self.is_invariant(constraint)
        
        target = jnp.array(target)
        vf = target if is_target_invariant else target[0]
        
        if constraint is not None:
            constraint = jnp.array(constraint)
            vf = jnp.maximum(vf, constraint if is_constraint_invariant else constraint[0])
        
        if constraint is None:
            def f(carry, j):
                i, _vf = carry
                _vf = hj.step(solver_settings, dynamics, grid, times[i], _vf, times[j+1])
                _vf = jnp.minimum(_vf, target if is_target_invariant else target[j+1])
                return (j, _vf), _vf
        else:
            def f(carry, j):
                i, _vf = carry
                _vf = hj.step(solver_settings, dynamics, grid, times[i], _vf, times[j+1])
                _vf = jnp.minimum(_vf, target if is_target_invariant else target[j+1])
                _vf = jnp.maximum(_vf, constraint if is_constraint_invariant else constraint[j+1])
                return (j, _vf), _vf

        if progress_bar:
            decorator = scan_tqdm(len(times)-1)
            f = decorator(f)
        
        if stack:
            return jnp.concatenate([
                vf[jnp.newaxis],
                jax.lax.scan(f, (0, vf), jnp.arange(len(times)-1))[1]
            ])
        else:
            return jax.lax.fori_loop(0, len(times) - 1, 
                                    lambda i, carry: f(carry, i)[0], 
                                    (0, vf))[1][jnp.newaxis]

    ## Auxiliary Methods ##

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

    def axis_index(self, ax: int | str, val: float):
        vec = self.axis_vec(ax)
        idx = jnp.abs(vec - val).argmin()
        if not jnp.isclose(vec[idx], val, atol=1e-5*self.axis_step(ax)):
            raise ValueError(f'Value {val} for axis {ax} ({self.axis_name(ax)}) is out of bounds {vec[[0,-1]]}')
        return idx

    def is_invariant(self, vf):
        return (True if vf is None else
                vf.shape[0] == 1 if len(vf.shape) == self.ndim else
                len(vf.shape) == len(self.grid.shape))

    def make_tube(self, vf):
        return (vf if not self.is_invariant(vf) else
                jnp.concatenate([vf[jnp.newaxis, ...]] * len(self.timeline)))

    ## Axes Interfaces ##

    def project_onto(
        self,
        vf: LevelSet,
        *,
        axes,
        keepdims: bool = False,
        union: bool = True,
        expand_full: bool = False,
        select: list[tuple[str, float]] | None = None,
    ):
        # Expect full-dimensional input before any selections
        if vf.ndim != self.ndim:
            raise ValueError(f"Expected {self.ndim}-D input, got {vf.ndim}-D")

        # Resolve kept axes (original indexing before selections)
        axes = [self.axis(ax) for ax in axes]
        if not (len(set(axes)) == len(axes) > 0):
            raise ValueError(f"No or duplicate axes in {axes}")

        # Collect (original_axis_index, key, value)
        select = ([] if select is None else 
                  [(self.axis(ax), val) for ax, val in select])
        # Apply in ascending order so we can track index shifts
        select.sort(key=lambda t: t[0])
        shift, removed = 0, []
        for i, val in select:
            if i in axes:
                raise ValueError(f"Cannot select axis {i} that is being projected onto")
            idx = self.axis_index(i, val)           # uses original axis semantics
            vf = jnp.take(vf, idx, axis=i-shift)    # adjust for previously removed axes
            removed.append(i)
            shift += 1
        # Adjust kept axes indices after removals
        axes = [a - sum(r < a for r in removed) for a in axes]

        # Determine which (current) axes to reduce
        reduce_dims = tuple(i for i in range(vf.ndim) if i not in axes)

        if union:
            out = vf.min(axis=reduce_dims, keepdims=keepdims)
        else:
            out = vf.max(axis=reduce_dims, keepdims=keepdims)

        if expand_full:
            # Expand to full kept-axis shape
            out = out * jnp.ones([n for i, n in enumerate(vf.shape) if i in axes])

        # Reorder kept axes to match user-specified order (currently sorted)
        current_order = sorted(axes)
        if current_order != axes:
            perm = [current_order.index(a) for a in axes]
            out = jnp.transpose(out, perm)

        return out

    ## Plotting Interfaces ##

    def plot(self, *args, **kwds):
        if 'method' not in kwds:
            kwds['method'] = 'isosurface'
        elif kwds['method'] == '2D':
            kwds['method'] = 'bitmap'
        elif kwds['method'] == '3D':
            kwds['method'] = 'isosurface'
        return super().plot(*args, **kwds)
    
    def transform_to_bitmap(self,
                             vf: LevelSet, *,
                             level: float = 0.0,
                             axes: tuple[Axis, Axis] = (0, 1),
                             **kwds):
        assert len(axes) == 2, "transform_to_bitmap expects exactly two axes"
        axes = tuple(self.axis(ax) for ax in axes)
        kwds.setdefault('expand_full', True)
        im = self.project_onto(vf, axes=axes, **kwds) <= level
        return im
    
    def transform_to_surface(self,
                              vf: LevelSet, *,
                              axes: tuple[Axis, Axis] = (0, 1),
                              **kwds):
        assert len(axes) == 2, "transform_to_surface expects exactly two axes"
        axes = tuple(self.axis(ax) for ax in axes)
        kwds.setdefault('expand_full', True)
        im = self.project_onto(vf, axes=axes, **kwds)
        return im

    def transform_to_isosurface(self,
                                vf: LevelSet, *,
                                axes: tuple[Axis, Axis, Axis] = (1, 2, 0), # time on z-axis by default
                                **kwds):
        assert len(axes) == 3, "transform_to_isosurface expects exactly three axes"
        axes = tuple(self.axis(ax) for ax in axes)
        kwds.setdefault('expand_full', True)
        im = self.project_onto(vf, axes=axes, **kwds)
        return im

    ## Set Interfaces ##        

    def halfspace(self, normal, offset, axes=None):
        axes = axes or list(range(self.ndim))
        axes = [self.axis(i) for i in axes]

        assert len(axes) == len(normal) == len(offset)
        
        data = jnp.zeros([self.shape[i] if i in axes else 1
                         for i in range(self.ndim)])
        for i, k, m in zip(axes, normal, offset):
            # Doesn't contribute
            if k == 0: continue
            
            # NOTE: Without this condition, problems may arise that are VERY ANNOYING to debug...
            amin, amax = self.axis_bounds(i)
            assert amin <= m <= amax, (
                f'For axis {i} ({self.axis_name(i)}): '
                f'Offset ({m}) must be inside boundary ({(amin, amax)}) to avoid numerical instability'
            )

            # normalize wrt discretization 
            # k /= self.axis_step(i)
            
            data -= k*( self.axis_vec(i) - m )
        
        return data

    def empty(self):
        # return jnp.ones(self._shape_inv)*jnp.inf # NOTE: something buggy
        return jnp.ones(self.shape)*jnp.inf
    
    def complement(self, vf):
        return jnp.asarray(-vf)
    
    def intersect(self, vf1, vf2):
        return jnp.maximum(vf1, vf2)

    def union(self, vf1, vf2):
        return jnp.minimum(vf1, vf2)
    
    def reachF(self, target, constraints=None):
        vf = self.solve(self.SOLVER_SETTINGS,
                        self.avoid_dynamics,
                        self.grid,
                        self.timeline,
                        target,
                        constraints,
                        stack=self._stack)
        return jnp.asarray(vf)

    def reach(self, target, constraints=None):
        if not self.is_invariant(target):
            target = jnp.flip(target, axis=0)
        if not self.is_invariant(constraints):
            constraints = jnp.flip(constraints, axis=0)
        vf = self.solve(self.SOLVER_SETTINGS,
                        self.reach_dynamics,
                        self.grid,
                        -self.timeline,
                        target,
                        constraints,
                        stack=self._stack)
        return jnp.flip(jnp.asarray(vf), axis=0)
    
    def avoid(self, target, constraints=None):
        if not self.is_invariant(target):
            target = jnp.flip(target, axis=0)
        if not self.is_invariant(constraints):
            constraints = jnp.flip(constraints, axis=0)
        vf = self.solve(self.SOLVER_SETTINGS,
                        self.avoid_dynamics,
                        self.grid,
                        -self.timeline,
                        target,
                        constraints,
                        stack=self._stack)
        return jnp.flip(jnp.asarray(vf), axis=0)

    def rci(self, target):
        return self.complement(self.avoid(self.complement(target)))

from secrets import token_hex

class TVHJImplDebugger(TVHJImpl):
    
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
    
    @staticmethod
    def _underline(s: str, char: str = '-') -> str:
        width = max(n for n in map(len, s.splitlines()))
        return char*width

    def is_invariant(self, inp):
        if type(inp) is tuple:
            vf, _ = inp
        else:
            vf = inp
        return super().is_invariant(vf)

    def halfspace(self, *args, **kwds):
        data = super().halfspace(*args, **kwds)
        print(s := f'{token_hex(2)} = PLANE{self._shape_str(data)}')
        print(self._underline(text := s) + '\n')
        return data, text

    def empty(self):
        out = super().empty()
        print(s := f'{token_hex(2)} = EMPTY{self._shape_str(out)}')
        print(self._underline(text := s) + '\n')
        return out, text

    def complement(self, inp):
        print(head := f'{token_hex(2)} = neg(')
        
        vf, s = inp
        print(s := self._indent(s) + ',')
        
        out = super().complement(vf)
        print(tail := f') => {self._shape_str(out)}')
        print(self._underline(text := '\n'.join([head, s, tail])) + '\n')
        return out, text

    def intersect(self, inp1, inp2):
        print(head := f'{token_hex(2)} = max(')

        vf1, s1 = inp1
        vf2, s2 = inp2

        print(s1 := self._indent(s1) + ',')
        print(s2 := self._indent(s2) + ',')

        out = super().intersect(vf1, vf2)
        print(tail := f') => {self._shape_str(out)}')
        print(self._underline(text := '\n'.join([head, s1, s2, tail])) + '\n')
        return out, text

    def union(self, inp1, inp2):
        print(head := f'{token_hex(2)} = min(')

        vf1, s1 = inp1
        vf2, s2 = inp2

        print(s1 := self._indent(s1) + ',')
        print(s2 := self._indent(s2) + ',')

        out = super().union(vf1, vf2)
        print(tail := f') => {self._shape_str(out)}')
        print(self._underline(text := '\n'.join([head, s1, s2, tail])) + '\n')
        return out, text
    
    def reach(self, target, constraints=None):
        print(head := f'{token_hex(2)} = reach(')

        vf1, s1 = target
        vf2, s2 = constraints

        print(s1 := self._indent(s1) + ',')
        print(s2 := self._indent(s2) + ',')

        out = super().reach(vf1, vf2)
        print(tail := f') => {self._shape_str(out)}')
        print(self._underline(text := '\n'.join([head, s1, s2, tail])) + '\n')
        return out, text

    def rci(self, inp):
        print(head := f'{token_hex(2)} = rci(')

        vf, s = inp

        print(s := self._indent(s) + ',')

        out = super().rci(vf)
        print(tail := f') => {self._shape_str(out)}')
        print(self._underline(text := '\n'.join([head, s, tail])) + '\n')
        return out, text