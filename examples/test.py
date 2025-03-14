# Generic TLT imports
from pyspect import *
from pyspect.langs.ltl import *
# Hybrid Zonotope imports
from hz_reachability.hz_impl import HZImpl
from hz_reachability.systems.cars import CarLinearModel2D, CarLinearModel4D
from hz_reachability.shapes import HZShapes
from hz_reachability.spaces import ParkingSpace

TLT.select(ContinuousLTL)

# Option 1: Use the existing set templates or cretate your own (Not implemented for HZ yet).
# e.g., state_space = ReferredSet('state_space')

# Option 2: Use the generic Set method to import any custom shape
shapes = HZShapes()
center = Set(shapes.center())
road_west = Set(shapes.road_west())
road_east = Set(shapes.road_east())
road_north = Set(shapes.road_north())
road_south = Set(shapes.road_south())

# Example task: Stay in road_e, or road_n UNTIL you REACH exit_n.
task = Until(Or(road_east, road_north), center)

reach_dynamics = CarLinearModel2D()

# Hybrid Zonotope implementation
impl = HZImpl(dynamics=reach_dynamics, space = ParkingSpace(), time_horizon = 5)

# Solve the problem - Find the states that can satisfy the task
out = TLT.construct(task).realize(impl)

# From string s, shift lines to the right by n spaces
def shift_lines(s, n):
    return '\n'.join([' '*n + l for l in s.split('\n')])
def print_hz(hz):
    dim = lambda m: "x".join(map(str, m.shape))
    print(f'Gc<{dim(hz.Gc)}>', shift_lines(str(hz.Gc), 2), sep='\n')
    print(f'Gb<{dim(hz.Gb)}>', shift_lines(str(hz.Gb), 2), sep='\n')
    print(f'C<{dim(hz.C)}>', shift_lines(str(hz.C), 2), sep='\n')
    print(f'Ac<{dim(hz.Ac)}>', shift_lines(str(hz.Ac), 2), sep='\n')
    print(f'Ab<{dim(hz.Ab)}>', shift_lines(str(hz.Ab), 2), sep='\n')
    print(f'b<{dim(hz.b)}>', shift_lines(str(hz.b), 2), sep='\n')

print_hz(out)

import hj_reachability as hj
import hj_reachability.shapes as shp

from pyspect.impls.hj_reachability import TVHJImpl
from hj_reachability.systems import Bicycle4D
from pyspect.plotting.levelsets import *

from math import pi

# Define origin and size of area, makes it easier to scale up/down later on 
X0, XN = -1.2, 2.4
Y0, YN = -1.2, 2.4
Z0, ZN = -1.2, 2.4

min_bounds = np.array([   X0,    Y0])
max_bounds = np.array([XN+X0, YN+Y0])
grid_space = (51, 51)

# min_bounds = np.array([   X0,    Y0,    Z0])
# max_bounds = np.array([XN+X0, YN+Y0, ZN+Z0])
# grid_space = (51,51,51)

# min_bounds = np.array([   X0,    Y0, -pi, 1.0])
# max_bounds = np.array([XN+X0, YN+Y0, +pi, 0.0])
# grid_space = (31, 31, 21, 11)

grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(min_bounds, max_bounds),
                                                               grid_space)

dynamics = dict(cls=None)

hj_impl = TVHJImpl(dynamics, grid, 3)
hj_impl.set_axes_names('t', 'x', 'y')

from scipy.ndimage import convolve
import numpy as np

def crop_mask(mask):
    """Reduce a binary mask to the smallest bounding box that includes all True values.

    Args:
        mask (ndarray): N-D boolean mask.

    Returns:
        cropped_mask (ndarray): Cropped version of the mask.
        slices (tuple): Tuple of slices that define the cropped region.
    """
    if not np.any(mask):  # Check if all False
        return mask, tuple(slice(0, 0) for _ in range(mask.ndim))  # Empty region
    
    # Find min/max indices for each axis
    slices = tuple(
        slice(np.min(indices), np.max(indices) + 1)
        for indices in (np.where(mask) if mask.ndim > 1 else (np.where(mask)[0],))
    )
    
    return mask[slices], slices

def generator(hz, i):
    Gc, Gb, C, Ac, Ab, b = hz
    nc = Ac.shape[0]
    ng = Ac.shape[1]

    C = C.reshape(-1)
    b = b.reshape(-1)
    
    # --- Continuous Constraints ---

    data = -np.inf * np.ones(grid.shape)
    for i in range(nc):
        data = np.max(
            data,
            shp.hyperplane(grid, normal=Ac[i], offset=[0]*ng, const=b[i]),
        )

    # --- Continuous Generators ---

    g = Gc[:, i:i+1]
    data = np.max(
        data,
        shp.intersection(
            shp.cylinder(grid, r=GEN_WIDTH, c=C, axis=g),
            shp.hyperplane(grid, normal=+g, offset=C+g),
            shp.hyperplane(grid, normal=-g, offset=C-g),
        ),
    )

def hz2hj(hz):

    Gc, Gb, C, Ac, Ab, b = hz
    ng = Gc.shape[1]

    ## Generators

    I = (generator(hz, 0) <= 0).astype(int)

    for i in np.arange(1, ng):
        K = (generator(hz, i) <= 0).astype(int)
        K = crop_mask(K)[0]

        I = convolve(I, K, mode='constant', cval=0) > 0

    vf = 0.5 - (I > 0)
    return vf

from scipy.signal import correlate
from tqdm import trange

GEN_WIDTH = np.linalg.norm(grid.spacings)

## ##

fig_select = 2
fig_kwds = dict(fig_theme='Light')

plot2D_bitmap(
    # M,
    **fig_kwds, 
    fig_enabled=fig_select==0,
) or \
plot3D_valuefun(
    # vf,
    min_bounds=min_bounds,
    max_bounds=max_bounds,
    **fig_kwds,
    fig_enabled=fig_select==1,
) or \
plot_levelsets(
    # shp.project_onto(vf, 1, 2),

    # (hz2hj(shapes.road_west()), dict(colorscale='blues')),
    # (hz2hj(shapes.road_east()), dict(colorscale='blues')),
    # (hz2hj(shapes.road_south()), dict(colorscale='blues')),
    # (hz2hj(shapes.road_north()), dict(colorscale='blues')),
    # (hz2hj(shapes.center()), dict(colorscale='greens')),

    (hz2hj(out.astuple()), dict(colorscale='greens')),

    # axes = (0, 1, 2),
    min_bounds=min_bounds,
    max_bounds=max_bounds,
    # plot_func=plot3D_levelset,
    **fig_kwds,
    fig_enabled=fig_select==2,
    fig_width=500, fig_height=500,
)