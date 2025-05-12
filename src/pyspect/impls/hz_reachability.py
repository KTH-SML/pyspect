import numpy as np
from scipy import sparse
import zonoopt as zono
import hj_reachability as hj

class hz_reachability:

    TIME_STEP = 0.02
    SOLVER_SETTINGS = zono.ADMM_settings() # default settings

    def __init__(self, dynamics, time_horizon, time_step=...):
        
        if time_step is Ellipsis: time_step = self.TIME_STEP
        self._dt = time_step

        Dynamics = dynamics.pop('cls')
        if Dynamics is not None:
            self.reach_dynamics = Dynamics(**dynamics).with_mode('reach')
            self.avoid_dynamics = Dynamics(**dynamics).with_mode('avoid')
        
        self.ndim = 2 # where to grab this?

    ## Auxiliary Methods ##
    @staticmethod
    def set_to_zono(S: hj.sets.BoundedSet, outer_approx=False, n_sides_approx=6):

        if isinstance(S, hj.sets.Box):
            # Box to Zono
            return zono.interval_2_zono(S.lo, S.hi)
        elif isinstance(S, hj.sets.Ball): # need to approximate
            if not S.center.shape[0] == 2:
                raise NotImplementedError("Ball to Zono not currently implemented for non-2D.")
            else:
                return zono.make_regular_zono_2D(S.radius, S.center, outer_approx, n_sides_approx)


    ## pyspect Interfaces ##
    def plane_cut(self, normal, offset, axes=None):
        # what does this method do?
        raise NotImplementedError("Plane cut not yet implemented.")

    def empty(self):
        # not sure if this is intended output of this method. Returning a zero-dimensional zonotope.
        return zono.Zono(sparse.csc_matrix(), np.array()) 
    
    def complement(self, Z):
        raise NotImplementedError("Complement is inefficient for hybrid zonotopes so not currently implemented.")
    
    def intersect(self, Z1, Z2):
        return zono.intersect(Z1, Z2)

    def union(self, Z1, Z2):
        # note: it is more efficient to do union_of_many([Z1, Z2, ..., ZN])
        # than to do union(union(Z1, Z2), Z3), etc.
        return zono.union_of_many([Z1, Z2]) 
    
    def reachF(self, target, constraints=None):
        pass # TO DO

    def reach(self, target, constraints=None):
        pass # TO DO