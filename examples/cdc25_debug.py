from pyspect import *

AXIS_NAMES = ['x', 'v'] # [m, m/s]
MAX_BOUNDS = [750,  30] # 500m, 30 mps ~= 110 kmph
MIN_BOUNDS = [  0,   0] #   0m,  0 mps
GRID_SHAPE = ( 91,  91)

MAX_ACCEL = 1.0     # [mps2]
TIME_STEP = 0.5     # [s]
TIME_HORIZON = 40   # [s]

# (1) Define the different regions.

HIGHWAY     = Union(BoundedSet(x=(..., 400), v=(15, ...)),   # highway
                    BoundedSet(x=(300, 420), v=(10,  20)))   # offramp
RESIDENTIAL = BoundedSet(x=(400, ...), v=( 5, 20))           # residential

# (2) Write the specification.

CITY = OR(HIGHWAY, RESIDENTIAL) # Use sets in place of propositions
TASK = UNTIL(CITY, 'home')

# (3) Select the set of primitive TLTs.

TLT.select(LTLc)

# (4) Create the TLT and set the proposition 'home'.

# Define the home region (implicit proposition in TASK).
H = BoundedSet(x=(700, 749), v=(1,  7), t=(TIME_HORIZON-2, ...))

objective = TLT(TASK).where(home=H)

# (5) Additional imports for the specific implementation.

from pyspect.impls.hj_reachability import TVHJImpl
from pyspect.plotting.levelsets import *

import hj_reachability as hj
from hj_reachability.systems import *

# (6) Define the implementation of the reachability algorithm.

dynamics = dict(cls=DoubleIntegrator,
                min_accel=-MAX_ACCEL,
                max_accel=+MAX_ACCEL)

impl = TVHJImpl(dynamics,
                AXIS_NAMES,
                MIN_BOUNDS,
                MAX_BOUNDS,
                GRID_SHAPE,
                TIME_HORIZON,
                time_step=TIME_STEP)

## (7) Run the reachability program ##

out = objective.realize(impl)

# `out` will have the same object type that `impl` operates with.
# For TVHJImpl, `out` will be a numpy array of the gridded value function.
print(f"{type(out) = }")

# (8) Plotting

plot3D_levelset(
    out,
    min_bounds=[           0, *MIN_BOUNDS],
    max_bounds=[TIME_HORIZON, *MAX_BOUNDS],
    xtitle='Position (m)',
    ytitle='Velocity (m/s)',
    eye=EYE_MH_W,
)

# Implementation independent

from time import time
from contextlib import contextmanager
from tqdm import tqdm

from pyspect import *

# HJ specific

from pyspect.impls.hj_reachability import TVHJImpl
from pyspect.plotting.levelsets import *

import hj_reachability as hj
from hj_reachability.systems import DoubleIntegrator as HJDoubleIntegrator

# HZ specific

from pyspect.plotting.zonotopes import _hz2hj

# from hz_reachability.hz_impl import TVHZImpl
from hz_reachability.systems.cars import *
from hz_reachability.systems.integrators import DoubleIntegrator as HZDoubleIntegrator
# from hz_reachability.spaces import EmptySpace
import pyspect.impls.hz_reachability as hz
import zonoopt as zono

@contextmanager
def timectx(msgfunc):
    """Context manager to time a block of code."""
    start = time()
    yield
    end = time()
    print(msgfunc(end-start))

AXIS_NAMES = ['x',  'v'] # [m, m/s]
MAX_BOUNDS = [+100, +20] # 500m, 30 mps ~= 110 kmph
MIN_BOUNDS = [-100, -20] #   0m,  0 mps
GRID_SHAPE = (  91,  91)

MAX_ACCEL = 1.0     # [mps2]
TIME_STEP = 0.5     # [s]
TIME_HORIZON = 40   # [s]

## SPECIFICATION

T = BoundedSet(x=(-50,  +50))

phi = ALWAYS(T)

# Define ALWAYS through RCI set (relating to ¬◇¬ψ)

@primitive(ALWAYS('_1'))
def Always_rci(_1: 'TLTLike') -> Tuple[SetBuilder, APPROXDIR]:
    b1, a1 = _1._builder, _1._approx

    ao = APPROXDIR.UNDER
    return (
        AppliedSet('rci', b1),
        ao + a1 if ao * a1 == APPROXDIR.EXACT else 
        a1      if ao == a1 else
        APPROXDIR.INVALID,
    )

# Define Always through fixed-point iteration

@primitive(ALWAYS('_1'))
def Always_fp(_1: 'TLTLike') -> Tuple[SetBuilder, APPROXDIR]:

    N = int(TIME_HORIZON / TIME_STEP)

    phi = 'psi'
    for _ in range(N-1):
        phi = AND('psi', NEXT(phi))

    tree = TLT(phi).where(psi=_1)
    return (tree._builder, tree._approx)

## CONSTRUCT TLT

TLT.select(LTLd | Always_fp)

tree = TLT(phi)

print(f"Approximation direction: {tree._approx = }")

## INITIALIZE IMPLEMENTATION

dynamics = HZDoubleIntegrator(max_accel=MAX_ACCEL, dt=TIME_STEP)
impl = hz.hz_reachability(dynamics,
                          AXIS_NAMES,
                          MIN_BOUNDS,
                          MAX_BOUNDS,
                          zono.interval_2_zono(zono.Box([-MAX_ACCEL], [+MAX_ACCEL])),
                          TIME_HORIZON,
                          time_step=TIME_STEP)

with timectx(lambda t: f"Realization with HZ took {t:.2f} seconds"):
    out = tree.realize(impl)

print(f"{type(out) = }")
print(f'n = {out.get_n()}, nGc = {out.get_nGc()}, nGb = {out.get_nGb()}, nC = {out.get_nC()}')

# plot
import matplotlib.pyplot as plt
zono.plot(out, color='b', edgecolor='b', alpha=0.5)
plt.axis('equal')
plt.show()