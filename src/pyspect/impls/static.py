"""Static backend implementations.

This module provides a concrete implementation for static sets. The goal
is to provide a simple base implementation that can be used for testing, debugging, and as a
reference for developing more complex backends. It also serves as a template for how to implement the
core interfaces for set operations.

Provided classes:
    - ``StaticImpl``: A simple implementation of the core set operations using NumPy arrays as level sets.
"""

import numpy as np

from .dev.axes import *
from .dev.plotly import *

__all__ = [
    'StaticImpl',
]

type LevelSet = np.ndarray


class StaticImpl(PlotlyImpl[LevelSet], AxesImpl[LevelSet]):

    def __init__(self, axes_spec):

        super().__init__(axes_spec)

        min_bounds = [spec['bounds'][0] for spec in axes_spec]
        max_bounds = [spec['bounds'][1] for spec in axes_spec]
        grid_shape = [(spec['points'] if 'points' in spec else
                       int(abs(max_bounds[i] - min_bounds[i]) // spec['step']) + 1)
                      for i, spec in enumerate(axes_spec)]

        self._dx = [(max_bounds[i] - min_bounds[i]) / (grid_shape[i] - 1) for i in range(self.ndim)]
        self._xx = [np.linspace(min_bounds[i], max_bounds[i], grid_shape[i]) for i in range(self.ndim)]
        self._shape = tuple(grid_shape)

        self.grid = np.stack(np.meshgrid(*self._xx, indexing='ij'), axis=-1)

    ## Auxiliary Methods ##

    def axis_step(self, ax: int | str):
        i = self.axis(ax)
        return self._dx[i]

    def axis_vec(self, ax: int | str):
        i = self.axis(ax)
        shape = tuple(n if j == i else 1
                      for j, n in enumerate(self._shape))
        return self._xx[i].reshape(shape)

    def axis_index(self, ax: int | str, val: float):
        vec = self.axis_vec(ax)
        idx = np.abs(vec - val).argmin()
        if not np.isclose(vec.take(idx), val, atol=self.axis_step(ax)/2):
            raise ValueError(f'Value {val} for axis {ax} ({self.axis_name(ax)}) is out of bounds {list(*vec.take([0, -1]))}')
        return idx
    
    def is_invariant(self, vf: LevelSet, ax: Axis | None = None):
        if ax is None:
            return np.array([vf.shape[i] == 1 for i in range(self.ndim)])
        else:
            i = self.axis(ax)
            return vf.shape[i] == 1

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
            idx = 0 if self.is_invariant(vf, i) else self.axis_index(i, val)
            vf = np.take(vf, idx, axis=i-shift)    # adjust for previously removed axes
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
            out = out * np.ones([n for i, n in enumerate(vf.shape) if i in axes])

        # Reorder kept axes to match user-specified order (currently sorted)
        current_order = sorted(axes)
        if current_order != axes:
            perm = [current_order.index(a) for a in axes]
            out = np.transpose(out, perm)

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
                                axes: tuple[Axis, Axis, Axis] = (0, 1, 2),
                                **kwds):
        assert len(axes) == 3, "transform_to_isosurface expects exactly three axes"
        axes = tuple(self.axis(ax) for ax in axes)
        kwds.setdefault('expand_full', True)
        im = self.project_onto(vf, axes=axes, **kwds)
        return im

    ## Set Interfaces ##        

    def empty(self):
        return np.ones(self._shape)*np.inf
    
    def complement(self, vf):
        return np.asarray(-vf)
    
    def intersect(self, vf1, vf2):
        return np.maximum(vf1, vf2)

    def union(self, vf1, vf2):
        return np.minimum(vf1, vf2)

    def halfspace(self, normal, offset, axes=None):
        axes = axes or list(range(self.ndim))
        axes = [self.axis(i) for i in axes]

        assert len(axes) == len(normal) == len(offset)
        
        data = np.zeros([self._shape[i] if i in axes else 1
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
