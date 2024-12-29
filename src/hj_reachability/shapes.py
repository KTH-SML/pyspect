from dataclasses import dataclass

import numpy as np
from .grid import Grid

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

def project_onto(vf, *idxs, keepdims=False):
    idxs = [len(vf.shape) + i if i < 0 else i for i in idxs]
    dims = [i for i in range(len(vf.shape)) if i not in idxs]
    return vf.min(axis=tuple(dims), keepdims=keepdims)

def hyperplane(grid: Grid, normal, offset, axes=None):
    """Creates an hyperplane implicit surface function

    Args:
        grid (Grid): Grid object
        normal (List): List specifying the normal of the hyperplane
        offset (float): offset of the hyperplane

    Returns:
        np.ndarray: implicit surface function of the hyperplane
    """
    data = np.zeros(grid.shape)
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

def cylinder(grid: Grid, center, axes, radius):
    """Creates an axis align cylinder implicit surface function

    Args:
        grid (Grid): Grid object
        ignore_dims (List): List specifing axis where cylinder is aligned (0-indexed)
        center (List): List specifying the center of cylinder
        radius (float): Radius of cylinder

    Returns:
        np.ndarray: implicit surface function of the cylinder
    """
    data = np.zeros(grid.shape)
    for i in range(grid.ndim):
        if i not in axes:
            # This works because of broadcasting
            data = data + np.power(grid.states[..., i] - center[i], 2)
    data = np.sqrt(data) - radius
    return data

def make_tube(times, vf):
    return np.concatenate([vf[np.newaxis, ...]] * len(times))

def is_invariant(grid, times, a):
    return a is None or len(a.shape) != len(times.shape + grid.shape)