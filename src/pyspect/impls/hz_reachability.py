import numpy as np
from scipy import sparse
import zonoopt as zono
import hj_reachability as hj

class hz_reachability:

    TIME_STEP = 0.02
    SOLVER_SETTINGS = zono.OptSettings() # default settings

    def __init__(self, dynamics, min_bounds: np.array, max_bounds: np.array, time_horizon: float, time_step=...):

        if time_step is Ellipsis: time_step = self.TIME_STEP
        self._dt = time_step

        Dynamics = dynamics.pop('cls')
        if Dynamics is not None:
            self.reach_dynamics = Dynamics(**dynamics).with_mode('reach')
            self.avoid_dynamics = Dynamics(**dynamics).with_mode('avoid')

        # state bounds
        self.Z_bounds = zono.interval_2_zono(min_bounds, max_bounds)
        
        self.ndim = 2 # where to grab this?

    ## Auxiliary Methods ##
    @staticmethod
    def set_to_zono(S: hj.sets.BoundedSet, outer_approx=False, n_sides_approx=6):
        """
        Converts a Hamilton-Jacobi bounded set to a zonotope.
        S: Hamilton-Jacobi bounded set
        outer_approx: if True, approximates the set with an outer zonotope, only used for Ball sets.
        n_sides_approx: number of sides for the outer approximation of a Ball set.
        """

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
        """
        Computes generalized halfspace intersection with bounds over specified axes.
        normal: array of normal vectors
        offset: array of offsets for the halfspaces
        axes: generalized intersection axes, if None, uses all axes
        """
        axes = axes or list(range(self.ndim))
        axes = [self.axis(i) for i in axes]

        assert len(axes) == len(normal) == len(offset)

        # generalized intersection matrix
        n_rows = len(axes)
        n_cols = self.Z_bounds.get_n()
        trip_rows = []
        trip_cols = []
        trip_values = []
        for i, ax in enumerate(axes):
            trip_rows.append(i)
            trip_cols.append(ax)
            trip_values.append(1)
        R = sparse.csc_matrix((trip_values, (trip_rows, trip_cols)), shape=(n_rows, n_cols))

        return zono.halfspace_intersection(self.Z_bounds, sparse.csc_matrix(normal), offset, R)

    def empty(self):
        # TO DO: return empty set of appropriate dimension
        pass
    
    def complement(self, Z):
        raise NotImplementedError("Need to implement complement.")
    
    def intersect(self, Z1, Z2):
        """
        Computes the intersection of two zonotopic sets.
        Z1, Z2: zonotopic sets to intersect
        """
        return zono.intersect(Z1, Z2)

    def union(self, Z1, Z2):
        """
        Computes the union of two zonotopic sets.
        Z1, Z2: zonotopic sets to union
        """
        return zono.union_of_many([Z1, Z2]) 
    
    def reachF(self, target, constraints=None):
        pass # TO DO

    def reach(self, target, constraints=None):
        pass # TO DO