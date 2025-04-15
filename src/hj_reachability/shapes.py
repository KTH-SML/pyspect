from dataclasses import dataclass

import numpy as np
from .grid import Grid

# If @ actually made sense for higher-order tensors
tmul = lambda x1, x2: np.tensordot(x1, x2, ([-1], [0]))

def complement(shape):
    """ Calculates the complement of a shape

    Args:
        shape (np.ndarray): implicit surface representation of a shape

    Returns:
        np.ndarray: the complement of the shape
    """
    return -shape

def union(shape, *shapes):
    """ Calculates the union of two shapes

    Args:
        shape1 (np.ndarray): implicit surface representation of a shape
        shape2 (np.ndarray): implicit surface representation of a shape

    Returns:
        np.ndarray: the element-wise minimum of two shapes
    """
    result = shape 
    for shape in shapes: 
        result = np.minimum(result, shape)
    return result

def intersection(shape, *shapes):
    """ Calculates the intersection of two shapes

    Args:
        shape1 (np.ndarray): implicit surface representation of a shape
        shape2 (np.ndarray): implicit surface representation of a shape

    Returns:
        np.ndarray: the element-wise minimum of two shapes
    """
    result = shape
    for shape in shapes:
        result = np.maximum(result, shape)
    return result

def setminus(a, *bs):
    result = a
    for b in bs:
        result = np.maximum(result, -b)
    return result

def project_onto(vf, *idxs, keepdims=False, union=True):
    idxs = [len(vf.shape) + i if i < 0 else i for i in idxs]
    dims = [i for i in range(len(vf.shape)) if i not in idxs]
    if union:
        return vf.min(axis=tuple(dims), keepdims=keepdims)
    else:
        return vf.max(axis=tuple(dims), keepdims=keepdims)


def hyperplane(grid: Grid, normal, offset, const=0, axes=None):
    """Creates an hyperplane implicit surface function

    Args:
        grid (Grid): Grid object
        normal (List): List specifying the normal of the hyperplane
        offset (float): offset of the hyperplane

    Returns:
        np.ndarray: implicit surface function of the hyperplane
    """
    data = -const * np.ones(grid.shape)
    axes = axes or list(range(grid.ndim))
    x = lambda i: grid.states[..., i]
    for i, k, m in zip(axes, normal, offset):
        data += k*x(i) - k*m
    return data

def lower_half_space(grid: Grid, axis, value):
    """Creates an axis aligned lower half space

    Args:
        grid (Grid): Grid object
        axis (int): Dimension of the half space (0-indexed)
        value (float): Used in the implicit surface function for V < value

    Returns:
        np.ndarray: implicit surface function of the lower half space
                    of size grid.pts_each_dim
    """
    normal = [0 if i != axis else 1 for i in range(grid.ndim)]
    offset = [0 if i != axis else value for i in range(grid.ndim)]
    return hyperplane(grid, normal, offset)

def upper_half_space(grid: Grid, axis, value):
    """Creates an axis aligned upper half space 

    Args:
        grid (Grid): Grid object
        axis (int): Dimension of the half space (0-indexed)
        value (float): Used in the implicit surface function for V > value

    Returns:
        np.ndarray: implicit surface function of the lower half space
                    of size grid.pts_each_dim
    """
    normal = [0 if i != axis else -1 for i in range(grid.ndim)]
    offset = [0 if i != axis else value for i in range(grid.ndim)]
    return hyperplane(grid, normal, offset)

def ranged_space(grid: Grid, axis, min_value, max_value):
    """Creates an axis aligned ranged space

    Args:
        grid (Grid): Grid object
        axis (int): Dimension of the half space (0-indexed)
        min_value (float): Used in the implicit surface function for V < min_value
        max_value (float): Used in the implicit surface function for V > max_value

    Returns:
        np.ndarray: implicit surface function of the lower half space
                    of size grid.pts_each_dim
    """
    return intersection(lower_half_space(grid, axis, max_value),
                        upper_half_space(grid, axis, min_value))

def rectangle(grid: Grid, target_min, target_max, axes=None):
    """Creates a rectangle implicit surface function

    Args:
        grid (Grid): Grid object
        target_min (List): List specifying the minimum corner of the rectangle
        target_max (List): List specifying the maximum corner of the rectangle
        axes (List): List specifying the axes of the rectangle

    Returns:
        np.ndarray: implicit surface function of the rectangle
    """
    periodics = grid._is_periodic_dim
    data = -np.inf * np.ones(grid.shape)
    if axes is None:
        axes = list(range(grid.ndim))
    elif isinstance(axes, int):
        axes = [axes]
    if isinstance(target_min, (int, float)):
        target_min = [target_min] * len(axes)
    if isinstance(target_max, (int, float)):
        target_max = [target_max] * len(axes)
    for i, vmin, vmax in zip(axes, target_min, target_max):
        if vmax < vmin and periodics[i]:
            patch = complement(ranged_space(grid, i, vmax, vmin))
        else:
            patch = ranged_space(grid, i, vmin, vmax)
        data = intersection(data, patch)
    return data

def point(grid: Grid, z):
    bounds = [(x - grid.spacings[i]/2, x + grid.spacings[i]/2)
              for i, x in enumerate(z)]
    target_min, target_max = zip(*bounds)
    return rectangle(grid, target_min=target_min, target_max=target_max)

def lp_bound(grid, c=..., r=1, w=..., p=2, axes=...):
    """
    = Creates a weighted Lp-bound shape.

    We define the Lp-bound shape as the set of points $x$ such that
    $ ( Sigma_(i in bb(N)_n) w_i | x_i - c_i |^p )^(1/p) <= r. $

    == Args

    / grid: Grid object defining $bb(R)^n$.
    / c: Center point of the Lp-bound. Default is 0.
    / r: The upper bound of the norm. Default is 1.
    / w: Weights for each dimension in the Lp norm. Default is 1.
    / p: Lp norm order. Default is 2.
    / axes: Defines indices $i$ by either:
            (1) if `= int(j)`, the single grid dimension $j$ s.t. $i in {j}$; or
            (2) if `= list(l)`, the list of grid dimensions s.t. $i in #`l`$; or
            (3) if `= array(A)`, redefine basis vectors using projection matrix
                $A in bb(R)^(n times m)$ s.t. $i in bb(N)_m$; or
            (4) if `= Ellipsis` (default), all grid dimensions are used.
    """

    match axes:
        case int(i):
            x = np.array([grid.states[..., i]])
        case list(l):
            x = np.array([grid.states[..., i] for i in l])
        case np.ndarray() as A:
            x = np.array([grid.states[..., i] for i in range(grid.ndim)])
            x = tmul(A, x)
        case Ellipsis:
            x = np.array([grid.states[..., i] for i in range(grid.ndim)])

    c = np.array([0] * len(axes) if c is ... else 
                 [c] if isinstance(c, (int, float)) else 
                 c)

    w = np.array([1] * len(axes) if w is ... else
                 [w] if isinstance(w, (int, float)) else 
                 w)

    assert len(c) == len(x), "Center point must have same dimension as x"
    assert len(w) == len(x), "Weights must have same dimension as x"
    
    # Needed for broadcasting
    c = c.reshape(-1, *[1]*grid.ndim)

    data = np.abs(x - c)**p
    data = tmul(w, data)
    data = np.power(data, 1/p) - r
    return data

def cylinder(grid, r, c, axis):
    """
    = Creates a cylinder implicit surface function

    == Args
    / grid: Grid object.
    / r: Radius of the cylinder.
    / c: Center of the cylinder (in grid space).
    / axis: Axis of the cylinder (in grid space).

    == Returns
    _(np.ndarray)_
    Implicit surface function of the cylinder.
    """
    match axis:
        case int(i):
            v = [0]*grid.ndim
            v[i] = 1
        case v: 
            v = np.array(v)
            v /= np.linalg.norm(v)
            assert len(c) == grid.ndim, "Center point must have same dimension as grid"
            assert len(v) == grid.ndim, "Axis must have same dimension as grid"

    # transformation matrix to space orthogonal to v
    # we're still in the same space, but have removed one axis.
    V = np.identity(grid.ndim) - np.outer(v, v)

    # move point c into the orthogonal space
    c = np.array([0] * len(axes) if c is ... else 
                 [c] if isinstance(c, (int, float)) else 
                 c)
    assert len(c) == len(v), "Center point must have same dimension as v"
    c = tmul(V, c)

    return lp_bound(grid, c=c, r=r, axes=V)
    
def make_tube(times, vf):
    return np.concatenate([vf[np.newaxis, ...]] * len(times))

def is_invariant(grid, times, a):
    return a is None or len(a.shape) != len(times.shape + grid.shape)