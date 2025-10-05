import numpy as np
from scipy import sparse
import zonoopt as zono
import hj_reachability as hj
from .axes import AxesImpl

class hz_reachability(AxesImpl):

    TIME_STEP = 0.02
    SOLVER_SETTINGS = zono.OptSettings() # default settings

    def __init__(self, dynamics, axis_names, min_bounds, max_bounds, input_set: zono.HybZono, time_horizon: float, time_step=...):
        super().__init__(axis_names, min_bounds, max_bounds)

        self.dynamics = dynamics

        if time_step is Ellipsis: time_step = self.TIME_STEP
        self._dt = time_step
        self.N = int(np.ceil(time_horizon / self._dt))

        # state and input sets
        self.S = zono.interval_2_zono(zono.Box(min_bounds, max_bounds))
        self.U = input_set
        self.nx = self.S.get_n()
        self.nu = self.U.get_n()

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
            return zono.interval_2_zono(zono.Box(S.lo, S.hi))
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
        axes = axes or list(range(self.nx))
        axes = [self.axis(i) for i in axes]

        assert len(axes) == len(normal) == len(offset)

        # H * x < = f, x in S
        H = -sparse.csc_matrix(normal)
        f = H*np.array(offset)

        # generalized intersection matrix
        n_rows = len(axes)
        n_cols = self.S.get_n()
        trip_rows = []
        trip_cols = []
        trip_values = []
        for i, ax in enumerate(axes):
            trip_rows.append(i)
            trip_cols.append(ax)
            trip_values.append(1)
        R = sparse.csc_matrix((trip_values, (trip_rows, trip_cols)), shape=(n_rows, n_cols))

        return zono.halfspace_intersection(self.S, H, f, R)

    def empty(self):
        """
        Returns a EmptySet object.
        """
        return zono.EmptySet(self.nx)
    
    def complement(self, Z: zono.HybZono):
        """
        Computes the set difference between the current zonotopic bounds and another zonotopic set.
        i.e., returns set_diff(S, Z)
        """
        return zono.set_diff(self.S, Z, settings=self.SOLVER_SETTINGS)
    
    def intersect(self, Z1: zono.HybZono, Z2: zono.HybZono):
        """
        Computes the intersection of two zonotopic sets.
        Z1, Z2: zonotopic sets to intersect
        """
        return zono.intersection(Z1, Z2)

    def union(self, Z1: zono.HybZono, Z2: zono.HybZono):
        """
        Computes the union of two zonotopic sets.
        Z1, Z2: zonotopic sets to union
        """
        return zono.union_of_many([Z1, Z2]) 
    
    def pre(self, target: zono.HybZono, constraints: zono.HybZono = None):
        """
        Computes one-step backward reachable set from target set
        """

        if constraints is None:
            constraints = self.S
        
        # get linear system matrices
        A = sparse.csc_matrix(self.dynamics.A)
        B = sparse.csc_matrix(self.dynamics.B)

        ABmI = sparse.hstack((A, B, -sparse.eye(self.nx)))

        Z = zono.cartesian_product(zono.cartesian_product(target, self.U), constraints)
        Z = zono.intersection(Z, zono.Point(np.zeros(self.nx)), ABmI)
        Z = zono.project_onto_dims(Z, [i for i in range(self.nx+self.nu, self.nx+self.nu+self.nx)])

        return Z

    def reachF(self, target: zono.HybZono, constraints: zono.HybZono = None):
        """
        Computes N-step forward reachable set from target set
        States are confined to the constraint set, which defaults to the zonotopic bounds.
        """
        if constraints is None:
            constraints = self.S
        
        # init reachable set
        Z = target

        # get linear system matrices
        A = sparse.csc_matrix(self.dynamics.A)
        B = sparse.csc_matrix(self.dynamics.B)
        ABmI = sparse.hstack((A, B, -sparse.eye(self.nx)))

        # loop through and compute reachable set
        for _ in range(self.N):
            Z = zono.cartesian_product(zono.cartesian_product(Z, self.U), constraints)
            Z = zono.intersection(Z, zono.Point(np.zeros(self.nx)), ABmI)
            Z = zono.project_onto_dims(Z, [i for i in range(self.nx+self.nu, self.nx+self.nu+self.nx)])

        # return N-step forwards reachable set
        return Z

    def reach(self, target: zono.HybZono, constraints: zono.HybZono = None):
        """
        Computes N-step backward reachable set from target set
        States are confined to the constraint set, which defaults to the zonotopic bounds.
        """
        if constraints is None:
            constraints = self.S
        
        # init reachable set
        Z = target

        # get linear system matrices
        A = self.dynamics.A
        B = self.dynamics.B
        Ainv = sparse.csc_matrix(np.linalg.inv(A))
        mAinvB = -Ainv * sparse.csc_matrix(B)
        Ainv_mAinvBmI = sparse.hstack((Ainv, mAinvB, -sparse.eye(self.nx)))

        # loop through and compute reachable set
        for _ in range(self.N):
            Z = zono.cartesian_product(zono.cartesian_product(Z, self.U), constraints)
            Z = zono.intersection(Z, zono.Point(np.zeros(self.nx)), Ainv_mAinvBmI)
            Z = zono.project_onto_dims(Z, [i for i in range(self.nx+self.nu, self.nx+self.nu+self.nx)])

        # return N-step backwards reachable set
        return Z