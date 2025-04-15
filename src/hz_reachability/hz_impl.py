# Python imports
import numpy as np
# Generic TLT imports
from pyspect import *
from pyspect.primitives.ltl import *
from pyspect.impls.axes import AxesImpl
# Hybrid Zonotope imports
from hz_reachability.auxiliary_operations import ZonoOperations
from hz_reachability.sets import HybridZonotope, ConstrainedZonotope
from copy import copy, deepcopy

"""
References
----------
[1] - Hybrid zonotopes: a new set representation for reachability analysis of mixed logical dynamical systems, Trevor J. Bird, et al.
[2] - Unions and Complements of Hybrid Zonotopes, Trevor J. Bird, Neera Jain.
[3] - Guaranteed Completion of Complex Tasks via Temporal Logic Trees and Hamilton-Jacobi Reachability, Frank J. Jiang, Kaj M. Arfvidsson, et al.
"""

class HZImpl(AxesImpl):
    """
    Description
    ------------
    This class contains all the supported methods for hybrid zonotope solution for TLTs within pyspect.

    Parameters
    ------------
    - dynamics
    - space: 
        - type: HybridZonotope
        - desc: The overarching state space within which the HZ computations take place.
    - time_horizon:
        - type: float
        - desc: The overall time duration in seconds of the reachability computations.
    - time_step:
        - type: float
        - desc: The discritization time step in seconds for the inbetween reachability computations.
    - show_intermediate:
        - type: Boolean
        - desc: Decides wether the intermediate steps of the reachability analysis should be returned back to the user or only the final one.
    """
    def __init__(self, dynamics, space, axis_names, time_horizon = 10, time_step = 0.1, show_intermediate = False):
        """
        Description
        ------------
        Implementing this method is NOT required by the pyspect API.
        Anything you implement here is set representation-specific.
        """
        self.dynamics = dynamics
        self.min_bounds = space.min_bounds
        self.max_bounds = space.max_bounds
        self.state_space = space.state_space
        self.input_space = dynamics.input_space
        self.augmented_space = self.augment_space()
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.N = int(self.time_horizon / self.time_step)
        self.has_disturbance = False
        self.show_intermediate = show_intermediate
        
        self.zono_op = ZonoOperations()
        self.enable_reduce = False

        super().__init__(axis_names, self.min_bounds, self.max_bounds)

    def empty(self):

        nz = self.state_space.dim
        ng = 0
        nb = 0
        nc = 0
        
        ## State Expression ##

        C = np.zeros((nz, 1))

        Gc = np.zeros((nz, ng))

        Gb = np.zeros((nz, nb))

        ## Constraints Expression ##

        Ac = np.zeros((nc, ng))

        Ab = np.zeros((nc, nb))

        b = np.zeros((nc, 1))

        return HybridZonotope(Gc = Gc, Gb = Gb, C = C, Ac = Ac, Ab = Ab, b = b)

    def complement(self, hz: HybridZonotope, constraint: HybridZonotope = None) -> HybridZonotope:
        """
        Description
        ------------        
        This method computes the complement of a hybrid Zonotope hz inside the 'constraint' space.
        The complement of the two hybrid zonotope is computed according to equation (16) in [2].
        
        In this particular implementation, to compute the complement, the hybrid zonotope, 
        is first over-approximated by a constrained zonotope. 
        Therefore, this method returns an under-approximation of the true complement.

        Parameters
        ------------        
        - hz: 
            - type: HybridZonotope
            - desc: The hybrid zonotope that its complement will be computed.
        - constraint: 
            - type: HybridZonotope
            - desc: The space that the hybrid zonotope and its complement should collectively form. Typically, this is the state space or a subscape of it.

        Returns
        ------------
        - constrained_compl: 
            - type: HybridZonotope
            - desc: The complement of the over-approximated (as a constrained zonotope) hybrid zonotope within the 'constraint' space.
        """
        if constraint is None:
            constraint = self.state_space
        
        # Catch empty case
        if hz.ng == hz.nb == 0:
            return self.state_space

        cz = self.oa_hz_to_cz(hz)
        
        G = cz.G; C = cz.C; A = cz.A; b = cz.b
        ng = cz.ng; nc = cz.nc; n = cz.dim
        ng_ng_zero = np.zeros((ng, ng))
        ng_1_zero  = np.zeros((ng, 1))
        ng_ng_eye  = np.eye(ng)
        ng_1_ones  = np.ones((ng, 1))
        ng_nnc_zero = np.zeros((ng, n + nc))
        ng_nncngng_zero = np.zeros((ng, n + nc + 2*ng))


        dm = 3000 # TODO
        lm = 3000 # TODO
        m = dm + 1


        AcPF = np.block([
            [ m*ng_ng_eye, -(dm/2)*ng_1_ones, ng_nncngng_zero],
            [-m*ng_ng_eye, -(dm/2)*ng_1_ones, ng_nncngng_zero]])

        AcDF = np.block([
            [       ng_ng_zero, ng_1_zero, lm*np.block([G.T, A.T]),        0.5*ng_ng_eye,       -0.5*ng_ng_eye],
            [np.zeros((1, ng)),         0,   np.zeros((1, n + nc)), 0.5*np.ones((1, ng)), 0.5*np.ones((1, ng))]])
        
        bDF = np.block([
            [ng_1_zero],
            [1 - ng]])

        AcCS = np.block([
            [-m*ng_ng_eye, (dm/2)*ng_1_ones, ng_nnc_zero, ng_ng_zero, ng_ng_zero],
            [ m*ng_ng_eye, (dm/2)*ng_1_ones, ng_nnc_zero, ng_ng_zero, ng_ng_zero],
            [  ng_ng_zero,        ng_1_zero, ng_nnc_zero,  ng_ng_eye, ng_ng_zero],
            [  ng_ng_zero,        ng_1_zero, ng_nnc_zero, ng_ng_zero,  ng_ng_eye]])

        AbCS = np.block([
            [m*ng_ng_eye,  ng_ng_zero],
            [ ng_ng_zero, m*ng_ng_eye],
            [ -ng_ng_eye,  ng_ng_zero],
            [ ng_ng_zero,  -ng_ng_eye]])

        cf1 = np.zeros((2*ng, 1))
        Gf1 = (m + dm/2) * np.eye(2*ng)

        cf2 = np.block([
            [-(dm + 1)*np.ones((2*ng, 1))],
            [-np.ones((2*ng, 1))]
        ])
        Gf2 = np.block([
            [((3*dm/2) + 1)*np.eye(2*ng), np.zeros((2*ng, 2*ng))],
            [np.zeros((2*ng, 2*ng))     , np.eye(2*ng)          ]
        ])


        Gc = np.block([m*G, np.zeros((n, 1+n+nc+ng+ng+2*ng+4*ng))])
        Gb = np.zeros((n, ng+ng))
        
        Ac = np.block([
            [m*A, np.zeros((nc, 1+n+nc+ng+ng+2*ng+4*ng))],
            [AcPF, Gf1, np.zeros((2*ng, 4*ng))],
            [AcDF, np.zeros((ng+1, 2*ng)), np.zeros((ng+1, 4*ng))],
            [AcCS, np.zeros((4*ng, 2*ng)), Gf2]
        ])

        Ab = np.block([
            [np.zeros((nc+2*ng + ng + 1, 2*ng))],
            [AbCS]
        ])

        b = np.block([
            [b],
            [cf1],
            [bDF],
            [cf2]
        ])

        compl = HybridZonotope(Gc, Gb, C, Ac, Ab, b)

        out = self.intersect(compl, constraint)

        if self.enable_reduce:
           out = self.reduce(out)
        return out
    
    def intersect(self, hz1: HybridZonotope, hz2: HybridZonotope) -> HybridZonotope:
        """
        Description
        ------------
        The intersection of the two hybrid zonotopes is computed according to equation (8) in [1].

        NOTE: This implementation assumes an identity transformation matrix along the intersection.

        For reference, it follows the mathematical definition of set intersection as given here:
        A intersect B = {x in F^{n} | x in A and x in B}, 
            - A, B subsets of R^{n}
            - F^{n} is the n-dimensional space of the field F

        Parameters
        ------------
        - hz1: 
            - type: HybridZonotope 
            - desc: The first set to be intersected
        - hz2: 
            - type: HybridZonotope
            - desc: The second set to be intersected

        Returns 
        ------------
        - intersect(hz1, hz2) 
            - type: HybridZonotope
            - desc: The intersection of the sets hz1 and hz2
        """
        # Quick checks
        if hz1 is self.state_space:
            return hz2
        if hz2 is self.state_space:
            return hz1

        C = hz1.C
        Gc = np.hstack( (hz1.Gc, np.zeros((hz1.dim, hz2.ng))) )
        Gb = np.hstack( (hz1.Gb, np.zeros((hz1.dim, hz2.nb))) )

        Ac = np.vstack((np.hstack( (hz1.Ac, np.zeros((hz1.nc, hz2.ng))) ),
                        np.hstack( (np.zeros((hz2.nc, hz1.ng)), hz2.Ac) ),
                        np.hstack( (hz1.Gc, -hz2.Gc))
                        ))
        Ab = np.vstack((np.hstack( (hz1.Ab, np.zeros((hz1.nc, hz2.nb))) ),
                        np.hstack( (np.zeros((hz2.nc, hz1.nb)), hz2.Ab) ),
                        np.hstack( (hz1.Gb, -hz2.Gb))
                        ))
        b = np.vstack((hz1.b, hz2.b, hz2.C - hz1.C))

        out = HybridZonotope(Gc, Gb, C, Ac, Ab, b)

        if self.enable_reduce:
           out = self.reduce(out)
        return out

    def union(self, hz1: HybridZonotope, hz2: HybridZonotope) -> HybridZonotope:
        """
        Description
        ------------
        The union between two hybrid zonotopes is computed according to [2]

        For reference, it follows the mathematical definition of set intersection as given here:
        A union B = {x in F^{n} | x in A or x in B}, 
            - A, B subsets of R^{n}
            - F^{n} is the n-dimensional space of the field F
        
        NOTE: This particular instance of the union of two hybrid zonotopes is a more elaborate implementation. That is done to reduce some additional computational complexity.        

        Parameters
        ------------
        - hz1: 
            - type: HybridZonotope 
            - desc: The first set to be unioned
        - hz2: 
            - type: HybridZonotope
            - desc: The second set to be unioned

        Returns
        ------------
        - Union(hz1, hz2): 
            - type: HybridZonotope 
            - desc: The union of the sets hz1 and hz2
        """
        
        # Catch empty case
        if hz1.ng == hz1.nb == 0:
            return hz2
        if hz2.ng == hz2.nb == 0:
            return hz1

        # Step 1: Solve the set of linear equations
        ones_1 = np.ones((hz1.nb, 1)); ones_2 = np.ones((hz2.nb, 1))
        Ab_1_ones = hz1.Ab @ ones_1; Ab_2_ones = hz2.Ab @ ones_2
        Gb_1_ones = hz1.Gb @ ones_1; Gb_2_ones = hz2.Gb @ ones_2

        Gb_hat = 0.5 * ( ( Gb_2_ones + hz1.C) - (Gb_1_ones + hz2.C) )
        Ab1_hat = 0.5 * ( -Ab_1_ones - hz1.b )
        Ab2_hat = 0.5 * ( Ab_2_ones + hz2.b )
        b1_hat = 0.5 * ( -Ab_1_ones + hz1.b )
        b2_hat = 0.5 * ( -Ab_2_ones + hz2.b )
        C_hat = 0.5 * ( ( Gb_2_ones + hz1.C) + (Gb_1_ones + hz2.C) )

        # Find the index of all non-zero columns of Gc
        nonzero_gc_1 = np.nonzero(hz1.Gc.any(axis=0))[0]
        nonzero_gb_1 = np.nonzero(hz1.Gb.any(axis=0))[0]
        nonzero_gb_1 = nonzero_gb_1# + hz1.ng

        # Find the index of all non-zero columns of Gc
        nonzero_gc_2 = np.nonzero(hz2.Gc.any(axis=0))[0]
        nonzero_gb_2 = np.nonzero(hz2.Gb.any(axis=0))[0]
        nonzero_gb_2 = nonzero_gb_2# + hz2.ng

        staircase_ac3_left  = np.zeros((nonzero_gc_1.shape[0], hz1.ng))
        staircase_ac3_right = np.zeros((nonzero_gc_2.shape[0], hz2.ng))

        for r, c in enumerate(nonzero_gc_1):
            staircase_ac3_left[r, c] = 1

        for r, c in enumerate(nonzero_gc_2):
            staircase_ac3_right[r, c] = 1


        staircase_ab3_left  = np.zeros((nonzero_gb_1.shape[0], hz1.nb))
        staircase_ab3_right = np.zeros((nonzero_gb_2.shape[0], hz2.nb))

        for r, c in enumerate(nonzero_gb_1):
            staircase_ab3_left[r, c] = 1
        for r, c in enumerate(nonzero_gb_2):
            staircase_ab3_right[r, c] = 1

        # Auxiliary variables
        n1 = 2*(staircase_ac3_left.shape[0] + staircase_ac3_right.shape[0] + staircase_ab3_left.shape[0] + staircase_ab3_right.shape[0])

        # Step 3: Construct the union of the hybrid zonotopes
        C = C_hat
        Gc = np.block([
            [hz1.Gc, hz2.Gc, np.zeros((hz1.dim, n1))],
        ])
        Gb = np.block([
            [hz1.Gb, hz2.Gb, Gb_hat]
        ])
        Ac3 = np.block([
            [                               staircase_ac3_left,    np.zeros((staircase_ac3_left.shape[0], hz2.ng))],
            [                              -staircase_ac3_left,    np.zeros((staircase_ac3_left.shape[0], hz2.ng))],
            [ np.zeros((staircase_ac3_right.shape[0], hz1.ng)),                                staircase_ac3_right],
            [ np.zeros((staircase_ac3_right.shape[0], hz1.ng)),                               -staircase_ac3_right],
            [  np.zeros((staircase_ab3_left.shape[0], hz1.ng)),    np.zeros((staircase_ab3_left.shape[0], hz2.ng))],
            [  np.zeros((staircase_ab3_left.shape[0], hz1.ng)),    np.zeros((staircase_ab3_left.shape[0], hz2.ng))],
            [ np.zeros((staircase_ab3_right.shape[0], hz1.ng)),   np.zeros((staircase_ab3_right.shape[0], hz2.ng))],
            [ np.zeros((staircase_ab3_right.shape[0], hz1.ng)),   np.zeros((staircase_ab3_right.shape[0], hz2.ng))]
        ])
        Ab3 = np.block([
            [  np.zeros((staircase_ac3_left.shape[0], hz1.nb)),  np.zeros((staircase_ac3_left.shape[0], hz2.nb)),   0.5*np.ones((staircase_ac3_left.shape[0], 1))],
            [  np.zeros((staircase_ac3_left.shape[0], hz1.nb)),  np.zeros((staircase_ac3_left.shape[0], hz2.nb)),   0.5*np.ones((staircase_ac3_left.shape[0], 1))],
            [ np.zeros((staircase_ac3_right.shape[0], hz1.nb)), np.zeros((staircase_ac3_right.shape[0], hz2.nb)), -0.5*np.ones((staircase_ac3_right.shape[0], 1))],
            [ np.zeros((staircase_ac3_right.shape[0], hz1.nb)), np.zeros((staircase_ac3_right.shape[0], hz2.nb)), -0.5*np.ones((staircase_ac3_right.shape[0], 1))],
            [                           0.5*staircase_ab3_left,  np.zeros((staircase_ab3_left.shape[0], hz2.nb)),   0.5*np.ones((staircase_ab3_left.shape[0], 1))],
            [                          -0.5*staircase_ab3_left,  np.zeros((staircase_ab3_left.shape[0], hz2.nb)),   0.5*np.ones((staircase_ab3_left.shape[0], 1))],
            [ np.zeros((staircase_ab3_right.shape[0], hz1.nb)),                          0.5*staircase_ab3_right, -0.5*np.ones((staircase_ab3_right.shape[0], 1))],
            [ np.zeros((staircase_ab3_right.shape[0], hz1.nb)),                         -0.5*staircase_ab3_right, -0.5*np.ones((staircase_ab3_right.shape[0], 1))]
        ])
        b3 = np.block([
            [  0.5*np.ones((staircase_ac3_left.shape[0], 1))],
            [  0.5*np.ones((staircase_ac3_left.shape[0], 1))],
            [ 0.5*np.ones((staircase_ac3_right.shape[0], 1))],
            [ 0.5*np.ones((staircase_ac3_right.shape[0], 1))],
            [     np.zeros((staircase_ab3_left.shape[0], 1))],
            [      np.ones((staircase_ab3_left.shape[0], 1))],
            [    np.zeros((staircase_ab3_right.shape[0], 1))],
            [     np.ones((staircase_ab3_right.shape[0], 1))],
        ])



        Ac = np.block([
            [         hz1.Ac           ,    np.zeros((hz1.nc, hz2.ng)),    np.zeros((hz1.nc, n1))],
            [np.zeros((hz2.nc, hz1.ng)),             hz2.Ac           ,    np.zeros((hz2.nc, n1))],
            [                           Ac3                           ,         np.eye(n1, n1)   ]
        ])

        Ab = np.block([
            [          hz1.Ab          ,    np.zeros((hz1.nc, hz2.nb)),    Ab1_hat],
            [np.zeros((hz2.nc, hz1.nb)),             hz2.Ab           ,    Ab2_hat],
            [                                 Ab3                                 ]
        ])

        b = np.block([
            [b1_hat],
            [b2_hat],
            [b3]
        ])       

        out = HybridZonotope(Gc, Gb, C, Ac, Ab, b)

        if self.enable_reduce:
           out = self.reduce(out)
        return out


    def reach(self, goal: HybridZonotope, constraint: HybridZonotope) -> HybridZonotope:
        """
        Description
        ------------
        This method is aligned with Definition ... in [3].
        It computes and returns the set of states the system can start at in the 'constraint' and is guaranteed it can reach the 'goal'

        NOTE: Reach: There exists a control input for all disturbance.
        It could work either with FRS or BRS.

        Parameters
        ------------
        - goal: 
            - type: HybridZonotope
            - desc: The target to be reached
        - constraint: 
            - type: HybridZonotope
            - desc: The space within which the reach set must lie. This is typically either the entire state space or a subspace of it.
                
        Returns
        ------------
        - reach_set: 
            - type: HybridZonotope
            - desc: reach set
        """
        if not self.has_disturbance:
            out = self.reach_no_disturbance(goal, constraint)
        else: 
            raise NotImplementedError

        if self.enable_reduce:
           out = self.reduce(out)
        return out

    
    def avoid(self, goal: HybridZonotope, constraint: HybridZonotope) -> HybridZonotope:
        """
        Description
        ------------
        NOTE: THIS METHOD HAS NOT BEEN VERIFIED YET THAT IT IS IN AGREEMENT WITH THE PYSPECT DEFINITIONS.

        Given the fynamics f(x, u, w) and space O that should be avoided, the avoid tube A, gives us the set of states the system can be in and avoid the danger zone O.
              
        NOTE: Avoid: For all control inputs there exists disturbance.
        It could work either with FRS or BRS

        Parameters
        ------------
        - goal: 
            - type: HybridZonotope
            - desc: The space to be avoided
        - constraint: 
            - type: HybridZonotope 
            - desc: The space within which the avoid set must lie. This is typically either the entire state space or a subspace of it.

        Returns
        ------------
        - avoid_set: 
            - type: HybridZonotope
            - desc: Avoid set
        """
        raise NotImplementedError()
    
    def plane_cut(self, normal, offset, axes=None, Z=None):

        nz = self.state_space.dim 
        axes = list(axes or range(nz))

        assert len(axes) == len(normal) == len(offset)

        _offset = np.zeros(nz)
        _offset[axes] = offset
        offset = _offset.reshape(nz, 1)

        _normal = np.zeros(nz)
        _normal[axes] = normal
        normal = _normal.reshape(nz, 1)

        # - Flip direction of normal, pyspect convention is pointing inwards.
        # - Normalize vector.
        normal *= -1 / np.linalg.norm(normal)

        if Z is None:
            Z = self.state_space
            
        rho = normal.T @ offset

        # naxes = [i for i in range(nz) if i not in axes]
        dm = (rho 
              - normal.T @ Z.C 
              + np.abs(normal.T @ Z.Gc).sum()
              + np.abs(normal.T @ Z.Gb).sum())
        
        Gc = np.block([
            [Z.Gc, np.zeros((Z.dim, 1))],
        ])
        Gb = Z.Gb
        C = Z.C

        Ac = np.block([
            [           Z.Ac, np.zeros((Z.nc, 1))], 
            [normal.T @ Z.Gc,                dm/2],
        ])
        Ab = np.block([
            [           Z.Ab],
            [normal.T @ Z.Gb],
        ])
        b = np.block([
            [Z.b],
            [rho - normal.T @ Z.C - dm/2],
        ])

        out = HybridZonotope(Gc, Gb, C, Ac, Ab, b)

        if self.enable_reduce:
           out = self.reduce(out)
        return out

    ############################################################################################################
    # Auxiliary Methods

    def reduce(self, hz):
        hz = self.zono_op.redundant_gc_hz(hz)
        hz = self.zono_op.redundant_c_hz(hz)
        return hz

    def augment_space(self, state_space=..., input_space=...):
        if state_space is Ellipsis:
            state_space = self.state_space
        if input_space is Ellipsis:
            input_space = self.input_space
        
        nz = state_space.Gc.shape[0]
        nu = input_space.Gc.shape[0]

        nz_c = state_space.Ac.shape[0]
        nz_b = state_space.Gb.shape[1]
        nz_g = state_space.Gc.shape[1]

        nu_c = input_space.Ac.shape[0]
        nu_b = input_space.Gb.shape[1]
        nu_g = input_space.Gc.shape[1]

        Gc  = np.block([[      state_space.Gc, np.zeros((nz, nu_g))],
                        [np.zeros((nu, nz_g)),       input_space.Gc]])
        Gb  = np.block([[      state_space.Gb, np.zeros((nz, nu_b))], 
                        [np.zeros((nu, nz_b)),       input_space.Gb]])
        c   = np.block([[state_space.C],
                        [input_space.C]])

        Ac  = np.block([[        state_space.Ac, np.zeros((nz_c, nu_g))], 
                        [np.zeros((nu_c, nz_g)),         input_space.Ac]])
        Ab  = np.block([[        state_space.Ab, np.zeros((nz_c, nu_b))], 
                        [np.zeros((nu_c, nz_b)),         input_space.Ab]])
        b   = np.block([[state_space.b], 
                        [input_space.b]])

        return HybridZonotope(Gc, Gb, c, Ac, Ab, b)


    def lt_hz(self, M: np.ndarray, hz: HybridZonotope) -> HybridZonotope:
        '''
        Computes the linear transformation of a hybrid zonotope
        A hybrid zonotope resultg from the linear transformation of a hybrid zonotope hz = (C, Gc, Gb, Ac, Ab, b)

        M @ hz = (M @ Gc, M @ Gb, M @ C, Ac, Ab, b)
        '''
        C = M @ hz.C
        Gc = M @ hz.Gc
        Gb = M @ hz.Gb
        Ac = hz.Ac
        Ab = hz.Ab
        b = hz.b

        return HybridZonotope(Gc, Gb, C, Ac, Ab, b)

    def ms_hz_hz(self, hz1: HybridZonotope, hz2: HybridZonotope) -> HybridZonotope:
        '''
        Computes the minkowski sum of two hybrid zonotopes.
        '''
        
        c = hz1.C + hz2.C

        Gc = np.block([
            hz1.Gc, hz2.Gc
        ])

        Gb = np.block([
            hz1.Gb, hz2.Gb
        ])

        Ac = np.block([
            [hz1.Ac, np.zeros((hz1.nc, hz2.ng))],
            [np.zeros((hz2.nc, hz1.ng)), hz2.Ac]
        ])

        Ab = np.block([
            [hz1.Ab, np.zeros((hz1.nc, hz2.nb))],
            [np.zeros((hz2.nc, hz1.nb)), hz2.Ab]
        ])

        b = np.block([
            [hz1.b], 
            [hz2.b]
        ])

        return HybridZonotope(Gc, Gb, c, Ac, Ab, b)

    def one_step_brs_hz_v2(self, X: HybridZonotope, U: HybridZonotope, T: HybridZonotope, A: np.ndarray, B: np.ndarray) -> HybridZonotope:
        BU = self.lt_hz(-B, U)
        T_plus_BU = self.ms_hz_hz(hz1 = T, hz2 = BU)
        A_inv = np.linalg.inv(A)
        A_inv_T_W_plus_BU = self.lt_hz(A_inv, T_plus_BU)

        # Compute intersection with safe space X
        X_intersection_A_inv_T_W_plus_BU = self.intersect(X, A_inv_T_W_plus_BU)

        return X_intersection_A_inv_T_W_plus_BU

    def predecessor(self, X: HybridZonotope, T: HybridZonotope, D: np.ndarray) -> HybridZonotope:
        """
        Description
        ------------
        Computes the predecessor (one-step backward reachable set)

        Parameters
        ------------
        - X
            - type: HybridZonotope
            - desc: Admissible set (state & control augmented space)
        - T
            - type: HybridZonotope
            - desc: Target set
        - D
            - type: np.ndarray
            - desc: The dynamics matrix (D = [A, B])

        Returns
        ------------
        predecessor
            - type: HybridZonotope
            - desc: The predecessor set
        """
        Gc = np.block([
            [X.Gc[:T.dim, :], np.zeros((T.dim, T.ng))]
        ])

        Gb = np.block([
            [X.Gb[:T.dim, :], np.zeros((T.dim, T.nb))]
        ])

        C = X.C[:T.dim, :]

        Ac = np.block([
            [         X.Ac         ,    np.zeros((X.nc, T.ng))],
            [np.zeros((T.nc, X.ng)),            T.Ac          ],
            [      D @ X.Gc        ,           -T.Gc          ]
        ])

        Ab = np.block([
            [         X.Ab         ,    np.zeros((X.nc, T.nb))],
            [np.zeros((T.nc, X.nb)),            T.Ab          ],
            [      D @ X.Gb        ,           -T.Gb          ]
        ])

        b = np.block([
            [X.b],
            [T.b],
            [T.C - D @ X.C]
        ])

        out = HybridZonotope(Gc, Gb, C, Ac, Ab, b)

        if self.enable_reduce:
           out = self.reduce(out)
        return out
    
    def reach_no_disturbance(self, goal: HybridZonotope, constraint: HybridZonotope) -> HybridZonotope:
        """
        Description
        ------------
        Computes the N-step backward reachable set, which in this case is equivalent to the reach set.

        Parameters
        ------------
        - goal
            - type: HybridZonotope
            - desc: The set of states the system needs to reach within a given time horizon
        - constraint
            - type: HybrisZonotope
            - desc: The space within which the reach set must lie. This is typically either the entire state space or a subspace of it.
        
        Returns
        ------------
        - goal
            - type: HybridZonotope
            - desc: The reach set
        """
        # TODO: Construct the augmented state space here based on the constraint and the input space if not already initialized during the creation of the object.
        all_predecessors = [deepcopy(goal)]
        for _ in range(self.N):
            goal = self.predecessor(self.augment_space(constraint), goal, self.dynamics.AB)
            all_predecessors.append(deepcopy(goal))
            
        if self.show_intermediate:
            return all_predecessors
        else:
            return goal
    
    def oa_hz_to_cz(self, hz: HybridZonotope) -> ConstrainedZonotope:
        """
        Description
        ------------
        This method takes in a hybrid zonotope and returns a constrained zonotope over-approximating the hybrid zonotope.

        Parameters
        ------------
        - hz
            - type: HybridZonotope
            - desc: The hybrid zonotope to be over-approximated as a constrained zonotope
        
        Returns
        ------------
        - cz
            - type: ConstrainedZonotope
            - desc: The over-approximated constrained zonotope
        """
        G = np.block([hz.Gc, hz.Gb])
        C = hz.C
        A = np.block([hz.Ac, hz.Ab])
        b = hz.b

        cz = ConstrainedZonotope(G, C, A, b)

        return cz
    
from tqdm import tqdm, trange

# Time-Variying Hybrid Zonotope
# If = HybridZonotope -> applies for entire time horizon
# If = list[HybridZonotope] -> correspond to time steps 
TVHZ = HybridZonotope | list[HybridZonotope]

class TVHZImpl(HZImpl):

    def __init__(self, dynamics, space, axis_names, time_horizon = 10, time_step = 0.1, show_intermediate = False):
        self.dynamics = dynamics
        self.min_bounds = space.min_bounds
        self.max_bounds = space.max_bounds
        self.state_space = space.state_space
        self.input_space = dynamics.input_space
        self.augmented_space = self.augment_space()
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.N = int(self.time_horizon / self.time_step)
        self.has_disturbance = False
        self.show_intermediate = show_intermediate
        
        self.zono_op = ZonoOperations()
        self.enable_reduce = False

        AxesImpl.__init__(
            self,
            (         't', *axis_names), 
            [           0, *space.min_bounds],
            [time_horizon, *space.max_bounds],
        )

    def empty(self) -> TVHZ:
        return super().empty()
    
    def complement(self, hz: TVHZ, space: TVHZ = None) -> TVHZ:
        if list not in (type(hz), type(space)):
            return super().complement(hz, space)
        if not isinstance(hz, list): hz = [hz] * self.N
        if not isinstance(space, list): space = [space] * self.N
        assert len(hz) == len(space) == self.N, 'Mismatching time length'
        return [super().complement(_hz, _space) for _hz, _space in zip(hz, space)]

    def intersect(self, hz1: TVHZ, hz2: TVHZ) -> TVHZ:
        if list not in (type(hz1), type(hz2)):
            return super().intersect(hz1, hz2)
        if not isinstance(hz1, list): hz1 = [hz1] * self.N
        if not isinstance(hz2, list): hz2 = [hz2] * self.N
        assert len(hz1) == len(hz2) == self.N, 'Mismatching time length'
        return [super().intersect(_hz1, _hz2) for _hz1, _hz2 in zip(hz1, hz2)]

    def union(self, hz1: TVHZ, hz2: TVHZ) -> TVHZ:
        if list not in (type(hz1), type(hz2)):
            return super().union(hz1, hz2)
        if not isinstance(hz1, list): hz1 = [hz1] * self.N
        if not isinstance(hz2, list): hz2 = [hz2] * self.N
        assert len(hz1) == len(hz2) == self.N, 'Mismatching time length'
        return [super().union(_hz1, _hz2) for _hz1, _hz2 in zip(hz1, hz2)]

    def reach(self, target: TVHZ, constr: TVHZ) -> TVHZ:
        # NOTE: currently requires constant target and constr
        #       we later evaluate as BRS
        assert not isinstance(target, list)
        assert not isinstance(constr, list)

        augm_space = self.augmented_space

        out = [super().intersect(target, constr)]

        for _ in trange(self.N-1):


            # pred = self.predecessor(augm_space, out[-1], self.dynamics.AB)


            _X = self.state_space
            _U = self.input_space
            _T = out[-1]
            _A = self.dynamics.A
            _B = self.dynamics.B

            pred = self.one_step_brs_hz_v2(_X, _U, _T, _A, _B)



            pred = super().intersect(pred, constr)
            out.append(pred)

        return out[::-1]

        ## TIME-VARYING INPUT ##

        # if not isinstance(target, list): target = [target] * self.N
        # if not isinstance(constr, list): constr = [constr] * self.N
        # assert len(target) == len(constr) == self.N, 'Mismatching time length'

        # out = []
        # pairs = list(zip(target, constr))

        # augm_space = self.augmented_space

        # enable_reduce = self.enable_reduce
        # self.enable_reduce = False

        # out.append(super().intersect(target[-1], constr[-1]))
        
        # for _target, _constr in tqdm(pairs[-2::-1]):
        #     pred = self.predecessor(augm_space, out[-1], self.dynamics.AB)
        #     pred = super().union(pred, _target)
        #     pred = super().intersect(pred, _constr)
        #     if enable_reduce:
        #         pred = self.reduce(pred)
        #     out.append(pred)

        # self.enable_reduce = enable_reduce
        # return out[::-1]

    def rci(self, goal: TVHZ) -> TVHZ:
        # # NOTE: currently requires constant target and constr
        # #       we later evaluate as BRS
        # assert not isinstance(goal, list)
        
        if isinstance(goal, list):
            goal = goal[::-1]
        else:
            goal = [goal] * self.N

        out = [goal[0]]

        for i in trange(1, self.N):

            _X = self.state_space
            _U = self.input_space
            _T = out[-1]
            _A = self.dynamics.A
            _B = self.dynamics.B

            pred = self.one_step_brs_hz_v2(_X, _U, _T, _A, _B)

            pred = super().intersect(pred, goal[i])
            out.append(pred)

        return out[::-1]

    def plane_cut(self, normal, offset, axes=None, **kwds):
        axes = axes or list(range(self.ndim))
        axes = [self.axis(i) for i in axes]
        assert len(axes) == len(normal) == len(offset), 'normal, offset and axes must be equal length'

        # remove these
        naxes = [i for i, k in zip(axes, normal) if k == 0]
        naxes.sort()
        while naxes:
            i = naxes.pop(-1)
            axes.pop(i)
            normal.pop(i)
            offset.pop(i)

        if 0 not in axes:
            return super().plane_cut(normal, offset, axes=[i-1 for i in axes], **kwds)
        else:
            assert len(axes) == 1, 'Not Implemented time-cut properly yet'
            i = axes[0]
            k = normal[0]
            m = offset[0]

            k *= -1 # pyspect convention is for normal to point into the set
            
            timeline = np.linspace(0, self.time_horizon, self.N)
            mask = k * (timeline - m) <= 0
            return [self.state_space if cond else self.empty()
                    for cond in mask]
