# Python imports
import numpy as np
# Generic TLT imports
from pyspect import *
from pyspect.langs.ltl import *
# Hybrid Zonotope imports
from hz_reachability.sets import HybridZonotope, ConstrainedZonotope
from copy import copy, deepcopy

"""
References
----------
[1] - Hybrid zonotopes: a new set representation for reachability analysis of mixed logical dynamical systems, Trevor J. Bird, et al.
[2] - Unions and Complements of Hybrid Zonotopes, Trevor J. Bird, Neera Jain.
[3] - Guaranteed Completion of Complex Tasks via Temporal Logic Trees and Hamilton-Jacobi Reachability, Frank J. Jiang, Kaj M. Arfvidsson, et al.
"""

class HZImpl(ContinuousLTL.Impl):
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
    def __init__(self, dynamics = None, space = None, time_horizon = 10, time_step = 0.1, show_intermediate = False):
        """
        Description
        ------------
        Implementing this method is NOT required by the pyspect API.
        Anything you implement here is set representation-specific.
        """
        self.dynamics = dynamics
        self.state_space = space.state_space if space is not None else None
        self.input_space = space.input_space if space is not None else None
        self.augmented_state_space = space.augmented_state_space if space is not None else None
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.N = int(self.time_horizon / self.time_step)
        self.has_disturbance = False
        self.show_intermediate = show_intermediate

    def complement(self, hz: HybridZonotope, constraint: HybridZonotope) -> HybridZonotope:
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

        constrained_compl = self.intersect(compl, constraint)

        return constrained_compl
    
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

        return HybridZonotope(Gc, Gb, C, Ac, Ab, b)

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

        return HybridZonotope(Gc, Gb, C, Ac, Ab, b)

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
            reach_set = self.reach_no_disturbance(goal, constraint)
        else: 
            raise NotImplementedError

        return reach_set
    
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
        # Step 1: Compute complement of the obstacle 'goal'.
        compl = self.complement(goal)

        # Step 2: Compute the BRS (solve reach problem) from the complement.
        if not self.has_disturbance:
            avoid_set = self.reach(compl, constraint)
        else:
            raise NotImplementedError

        return avoid_set
    

    ############################################################################################################
    # Auxiliary Methods

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

        predecessor = HybridZonotope(Gc, Gb, C, Ac, Ab, b)

        return predecessor
    
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
            goal = self.predecessor(self.augmented_state_space, goal, self.dynamics.AB)
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
    

