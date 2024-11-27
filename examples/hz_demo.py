# Generic TLT imports
from pyspect import *
from pyspect.langs.ltl import *
# Hybrid Zonotope imports
from hz_reachability.hz_impl import HZImpl
from hz_reachability.systems.cars import CarLinearModel2D
from hz_reachability.shapes import HZShapes
from hz_reachability.spaces import ParkingSpace

# ########################################################################################
# Lazy evaluation: Check if the TLT structure support the continuous LTL.
# ########################################################################################
TLT.select(ContinuousLTL)

# ########################################################################################
# Environment definition
# ########################################################################################

# Option 1: Use the existing set templates or cretate your own (Not implemented for HZ yet).
# e.g., state_space = ReferredSet('state_space')

# Option 2: Use the generic Set method to import any custom shape
shapes = HZShapes()
center = Set(shapes.center())
road_west = Set(shapes.road_west())
road_east = Set(shapes.road_east())
road_north = Set(shapes.road_north())
road_south = Set(shapes.road_south())

# ########################################################################################
# Task Definition
# ########################################################################################

# Example task: Stay in road_e, or road_n UNTIL you REACH exit_n.
task = Until(Or(road_east, road_north), center)

# ########################################################################################
# Dynamics
# ########################################################################################
reach_dynamics = CarLinearModel2D()

# ########################################################################################
# Choose Reachability Analysis Implementation (e.g., HZ, HJ) 
# ########################################################################################

# Hybrid Zonotope implementation
impl = HZImpl(dynamics=reach_dynamics, space = ParkingSpace(), time_horizon = 5)

# ########################################################################################
# Solve the problem: # Find the states that can satisfy the task
# ########################################################################################

out = TLT.construct(task).realize(impl)

print(f'out = {out}')
print(f'Gc = \n{out.Gc}')

# ########################################################################################
# Additional Notes:
# ########################################################################################
'''
construct(task): Take an LTL, a set, or a lazy set, or an already constructed TLT and make 
sure it is a valid TLT object. Basically construct the compute graph for the given task.

realize(impl), initiates the actual computations.
out: The final set in your specific set implementation.
e.g., it would be a hybrid zonotope.
'''
