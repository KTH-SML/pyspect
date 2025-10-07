import numpy as np
import zonoopt as zono
from scipy import sparse
import warnings

from .axes import *
from .plotly import *

type Axes = tuple[Axis, ...]

# TODO: Move
class DoubleIntegrator:
    
    dt = 0.05

    def __init__(self, max_accel, dt=dt) -> None:
        
        ## Dynamics ##

        self.A = np.array([
            [1.0,  self.dt],
            [0.0, 1.0],
        ])
        self.B = np.array([
            [self.dt**2/2],
            [self.dt     ],
            # [0],
            # [1],
        ]) * max_accel


class ZonoOptImpl(PlotlyImpl[zono.HybZono], AxesImpl[zono.HybZono]):

    TIME_STEP = 0.02
    SOLVER_SETTINGS = zono.OptSettings() # default settings

    def __init__(self, dynamics, axes, input_set: zono.HybZono, time_horizon: float, time_step=...):
        
        # Initialize AxesImpl
        super().__init__(axes)

        self.dynamics = dynamics

        if time_step is Ellipsis: time_step = self.TIME_STEP
        self._dt = time_step
        self.N = int(np.ceil(time_horizon / self._dt))

        # state and input sets
        self.S = zono.interval_2_zono(zono.Box(self._min_bounds, self._max_bounds))
        self.U = input_set
        self.nx = self.S.get_n()
        self.nu = self.U.get_n()

    def transform_to_scatter(self, inp: zono.HybZono, *, axes: Axes = (0, 1, 2), t_max=60.0, settings=zono.OptSettings()):
        """Transforms zonotopic set to scatter plot data.
        inp: zonotopic set to be transformed
        axes: axes to plot on
        t_max: maximum time to spend on finding vertices
        settings: optimization settings
        """

        if inp.get_n() < 2 or inp.get_n() > 3:
            raise ValueError("Plot only implemented in 2D or 3D")
        
        # hybzono -> get leaves
        if inp.is_hybzono():
            raise NotImplementedError('transform_to_scatter not implemented for HybZono')

        return zono.get_vertices(inp, t_max=t_max)

    ## pyspect Interfaces ##
    def halfspace(self, normal, offset, axes=None):
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