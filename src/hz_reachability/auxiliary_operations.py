import math
import numpy as np

# For mixed-integer linear programming
import gurobipy as gp
from gurobipy import *

from hz_reachability.sets import HybridZonotope, ConstrainedZonotope

class ZonoOperations:
    """
    Description
    ------------
    This class contains all the required auxiliary methods for hybrid zonotope spaces.
    For examplle it includes methods that help with defining hybrid zonotope spaces, performing order reductions, etc.
    """
    def __init__(self):
        pass

    def union(self, hz1: HybridZonotope, hz2: HybridZonotope) -> HybridZonotope:
        """
        Description
        ------------
        The union between two hybrid zonotopes is computed according to [2].
        A union B = {x in F^{n} | x in A or x in B}, 
            - A, B subsets of R^{n}
            - F^{n} is the n-dimensional space of the field F
        
        NOTE: This particular instance of the union of two hybrid zonotopes is a more elaborate implementation. That is done to reduce some additional computational complexity.        

        Parameters
        ------------
        - hz1: HybridZonotope 
            - The first set to be unioned
        - hz2: HybridZonotope
            - The second set to be unioned

        Returns
        ------------
        - Union(hz1, hz2): HybridZonotope 
            - The union of the sets hz1 and hz2
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

        # Step 3: Construst the union of the hybrid zonotopes
        C = C_hat
        Gc = np.block([
            [hz1.Gc, hz2.Gc, np.zeros((hz1.dim, n1))],
        ])
        Gb = np.block([
            [hz1.Gb, hz2.Gb, Gb_hat]
        ])
        Ac3 = np.block([
            [     staircase_ac3_left,   np.zeros((staircase_ac3_left.shape[0], hz2.ng))],
            [    -staircase_ac3_left,   np.zeros((staircase_ac3_left.shape[0], hz2.ng))],
            [ np.zeros((staircase_ac3_right.shape[0], hz1.ng)),       staircase_ac3_right],
            [ np.zeros((staircase_ac3_right.shape[0], hz1.ng)),      -staircase_ac3_right],
            [ np.zeros((staircase_ab3_left.shape[0], hz1.ng)),   np.zeros((staircase_ab3_left.shape[0], hz2.ng))],
            [ np.zeros((staircase_ab3_left.shape[0], hz1.ng)),   np.zeros((staircase_ab3_left.shape[0], hz2.ng))],
            [ np.zeros((staircase_ab3_right.shape[0], hz1.ng)),   np.zeros((staircase_ab3_right.shape[0], hz2.ng))],
            [ np.zeros((staircase_ab3_right.shape[0], hz1.ng)),   np.zeros((staircase_ab3_right.shape[0], hz2.ng))]
        ])
        Ab3 = np.block([
            [ np.zeros((staircase_ac3_left.shape[0], hz1.nb)),  np.zeros((staircase_ac3_left.shape[0], hz2.nb)), 0.5*np.ones((staircase_ac3_left.shape[0], 1))],
            [ np.zeros((staircase_ac3_left.shape[0], hz1.nb)),  np.zeros((staircase_ac3_left.shape[0], hz2.nb)), 0.5*np.ones((staircase_ac3_left.shape[0], 1))],
            [ np.zeros((staircase_ac3_right.shape[0], hz1.nb)), np.zeros((staircase_ac3_right.shape[0], hz2.nb)), -0.5*np.ones((staircase_ac3_right.shape[0], 1))],
            [ np.zeros((staircase_ac3_right.shape[0], hz1.nb)), np.zeros((staircase_ac3_right.shape[0], hz2.nb)), -0.5*np.ones((staircase_ac3_right.shape[0], 1))],
            [ 0.5*staircase_ab3_left,    np.zeros((staircase_ab3_left.shape[0], hz2.nb)),     0.5*np.ones((staircase_ab3_left.shape[0], 1))],
            [-0.5*staircase_ab3_left,    np.zeros((staircase_ab3_left.shape[0], hz2.nb)),     0.5*np.ones((staircase_ab3_left.shape[0], 1))],
            [ np.zeros((staircase_ab3_right.shape[0], hz1.nb)),    0.5*staircase_ab3_right,    -0.5*np.ones((staircase_ab3_right.shape[0], 1))],
            [ np.zeros((staircase_ab3_right.shape[0], hz1.nb)),   -0.5*staircase_ab3_right,    -0.5*np.ones((staircase_ab3_right.shape[0], 1))]
        ])
        b3 = np.block([
            [0.5*np.ones((staircase_ac3_left.shape[0], 1))],
            [0.5*np.ones((staircase_ac3_left.shape[0], 1))],
            [0.5*np.ones((staircase_ac3_right.shape[0], 1))],
            [0.5*np.ones((staircase_ac3_right.shape[0], 1))],
            [    np.zeros((staircase_ab3_left.shape[0], 1))],
            [     np.ones((staircase_ab3_left.shape[0], 1))],
            [    np.zeros((staircase_ab3_right.shape[0], 1))],
            [     np.ones((staircase_ab3_right.shape[0], 1))],
        ])



        Ac = np.block([
            [         hz1.Ac           ,    np.zeros((hz1.nc, hz2.ng)),    np.zeros((hz1.nc, n1))],
            [np.zeros((hz2.nc, hz1.ng)),             hz2.Ac           ,    np.zeros((hz2.nc, n1))],
            [                                 Ac3                     ,         np.eye(n1, n1)   ]
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

    ############################################################################################################
    # Redundancey removal methods: Be very cautious when using them.

    def redundant_c_gc_hz_v2(self, hz: HybridZonotope) -> HybridZonotope:
        '''
        # This version does not always work. 
        Nevertheless, I still use it for the initial space as it works in these cases and reduces the redundancy better than redundant_c_gc_hz_v1
        '''
        E, R = self.intervals_hz(hz)
        A = hz.Ac

        epsilon = 1e-3

        already_removed_c = []; already_removed_g = []
        for c in range (hz.ng):
            for r in range(hz.nc):
                if np.abs(A[r, c]) >= epsilon:
                    a_rc_inv = (1/A[r, c])
                    sum = 0
                    for k in range(hz.ng):
                        if k != c:
                            sum = sum + A[r, k] * E[k]
                    R_rc = a_rc_inv * hz.b[r,0] - a_rc_inv * sum

                    if self.is_inside_interval(R_rc, np.array([-1, 1])) and (r not in already_removed_c) and (c not in already_removed_g):
                        already_removed_c.append(r); already_removed_g.append(c)
                        Ecr = np.zeros((hz.ng, hz.nc))
                        Ecr[c, r] = 1

                        Lg = hz.Gc @ Ecr * (1/A[r, c])
                        La = hz.Ac @ Ecr * (1/A[r, c])

                        # Check if Lg has only zero zero values
                        full_zero = True
                        for x in range(Lg.shape[1]):
                            for y in range(Lg.shape[0]):
                                if Lg[y, x] != 0:
                                    full_zero = False

                        if not (full_zero):
                            Gc = hz.Gc - Lg @ hz.Ac
                            Gb = hz.Gb - Lg @ hz.Ab
                            C  = hz.C  + Lg @ hz.b
                            Ac = hz.Ac - La @ hz.Ac
                            Ab = hz.Ab - La @ hz.Ab
                            b  = hz.b  - La @ hz.b

                            hz = HybridZonotope(Gc, Gb, C, Ac, Ab, b)



        hz = self.reduce_c_hz(hz)   # Remove the redundant or zero constraints
        hz = self.reduce_gc_hz(hz)

        return hz    
    
    def reduce_c_hz(self, hz: HybridZonotope) -> HybridZonotope:
        '''
        Reduces the number of constraints of a Hybrid Zonotope.

        In this version we are removing the following constraints:
            - Any constraint whose constraint matrix component (the particular row in [Ac Ab]) is all zeros
            - Any constraint that there is another constraint that is equivalent to it
                e.g., x + y = 1 and 2x + 2y = 2, 5x + 5x = 5 only one out of these three constraints will be kept

        TODO: NOW THAT YOU HAVE IMPLEMENTED THE NEW METHOD CHECK IF THIS IS REDUNDANT
        '''
        max_angle = 0.05 * math.pi / 180
        threshold = 1 - math.sin(max_angle)

        A = np.block([hz.Ac, hz.Ab, hz.b])

        nc = A.shape[0]

        # Loop through all the columns of Gc
        i = 0; j = 0; k = 0

        while i < nc - k:
            c1 = A[i, :].T
            c1_mag = np.linalg.norm(c1)    # Magnitude of c1

            if np.abs(c1_mag) <= 0.001:
                A = np.delete(A, i, axis=0)
                k += 1
                continue

            c1_unit = c1 / c1_mag           # Unit vector of c1


            j = 0
            while j < nc - k:
                if i == j:
                    j += 1
                    continue

                c2 = A[j, :].T
                c2_mag = np.linalg.norm(c2)     # Magnitude of c2

                if (c2_mag <= 0.001) or np.abs(np.dot(c1_unit.T, c2 / c2_mag)) >= threshold:
                    A = np.delete(A, j, axis=0)   # Remove the second constraint
                    k += 1

                j += 1

            i +=1

        Ac = A[:, :hz.Ac.shape[1]]
        Ab = A[:, hz.Ac.shape[1]:-1]
        b = A[:, -1].reshape((A.shape[0], 1))

        return HybridZonotope(hz.Gc, hz.Gb, hz.C, Ac, Ab, b)

    def reduce_gc_hz(self, hz: HybridZonotope) -> HybridZonotope:
        '''
        Removes redundant continuous generators from a Hybrid Zonotope.

        This method first forms the lifted hybrid zonotope. Then it
        scans all generators and whenver it finds a pair of generators that are
        parallel to each other, it adds one to the other and removes the other one.

        Example: If we have two generators g1 and g2, and g1 = 2*g2,
        then we update g1 as g1 = g1 + g2 = 3*g2 and remove g2. 

        TODO: NOW THAT YOU HAVE IMPLEMENTED THE NEW METHOD CHECK IF THIS IS REDUNDANT

        '''

        # threshold = 1e-7
        max_angle = 0.05 * math.pi / 180
        # max_angle = 0.5 * math.pi / 180
        threshold = 1 - math.sin(max_angle)

        # Step 1: Stack Gc and Ac
        G = np.block([
            [hz.Gc],
            [hz.Ac]
        ])

        ng = G.shape[1]

        # Loop through all the rows of Gc
        i = 0; j = 0; k = 0

        while i < ng - k:
            g1 = G[:, i]
            g1_mag = np.linalg.norm(g1)    # Magnitude of g1

            if np.abs(g1_mag) <= 0.001:
                G = np.delete(G, i, axis=1)
                k += 1
                continue

            g1_unit = g1 / g1_mag           # Unit vector of g1


            j = 0
            while j < ng - k:
                if i == j:
                    j += 1
                    continue

                g2 = G[:, j]
                g2_mag = np.linalg.norm(g2)     # Magnitude of g2


                if (g2_mag <= 0.001):
                    G = np.delete(G, j, axis=1)   # Remove the second generator
                    k += 1                    
                # elif np.abs(np.dot(g1_unit.T, g2 / g2_mag)) >= threshold:
                #     G[:, i - k] = g1 + g2
                #     G = np.delete(G, j, axis=1)   # Remove the second generator
                #     k += 1

                j += 1

            i +=1

        Gc = G[:hz.dim, :]
        Ac = G[hz.dim:, :]


        return HybridZonotope(Gc, hz.Gb, hz.C, Ac, hz.Ab, hz.b)

    def redundant_c_gc_hz_v1(self, hz: HybridZonotope, options = 'slow') -> HybridZonotope:
        '''
        This method performs redundancy removal for constraints and continuous generators in a hybrid zonotope.

        # This version always works
        '''
        epsilon = 1e-3
        redundant = True

        Eb = np.array([ [-1, 1] for b in range(hz.nb) ])

        hz = self.redundant_c_hz(hz)
        hz = self.redundant_gc_hz(hz)

        while redundant:
            redundant = False
            hz = self.rref_hz(hz)
            if options == 'fast':
                E = self.find_E_hz_fast(hz)
            else:
                E = self.find_E_hz_slow(hz)

            for c in range (hz.ng):
                for r in range(hz.nc):
                    if np.abs(hz.Ac[r, c]) >= epsilon:
                        a_rc_inv = (1/hz.Ac[r, c])
                        R_rc = np.array([hz.b[r,0], hz.b[r,0]])

                        tempc = np.array([0.0, 0.0])
                        tempb = np.array([0.0, 0.0])
                        for k in range(hz.ng):
                            if k != c:
                                tempc = self.interval_add(tempc, self.interval_scalar_mul(hz.Ac[r, k], E[k, :]))

                        for b in range(hz.nb):
                            tempb = self.interval_add(tempb, self.interval_scalar_mul(hz.Ab[r, b], Eb[b, :]))

                        R_rc = self.interval_sub(R_rc, tempc)
                        R_rc = self.interval_sub(R_rc, tempb)

                        R_rc = self.interval_scalar_mul(a_rc_inv, R_rc)

                        if self.is_inside_interval(R_rc, np.array([-1, 1])):
                            hz = self.remove_c_g_hz(hz = hz, c = c, r = r)
                            redundant = True

                            break
                if redundant:
                    break

        return hz
    
    def remove_c_g_hz(self, hz, c, r):
        '''
        c: constraint index
        r: generator index
        '''
        Ecr = np.zeros((hz.ng, hz.nc))
        Ecr[c, r] = 1

        Lg = hz.Gc @ Ecr * (1/hz.Ac[r, c])
        La = hz.Ac @ Ecr * (1/hz.Ac[r, c])

        Gc = hz.Gc - Lg @ hz.Ac
        Gb = hz.Gb - Lg @ hz.Ab
        C  = hz.C  + Lg @ hz.b
        Ac = hz.Ac - La @ hz.Ac
        Ab = hz.Ab - La @ hz.Ab
        b  = hz.b  - La @ hz.b

        Gc = np.delete(Gc, c, axis=1)
        Ac = np.delete(Ac, c, axis=1)
        Ac = np.delete(Ac, r, axis=0)
        Ab = np.delete(Ab, r, axis=0)
        b  = np.delete(b, r, axis=0)

        return HybridZonotope(Gc, Gb, C, Ac, Ab, b)

    def redundant_gc_hz(self, hz: HybridZonotope) -> HybridZonotope:
        '''
        Removes redundant generators from a Hybrid Zonotope.

        This method first partially forms the lifted hybrid zonotope. Then it
        scans all generators and whenver it finds a pair of generators that are
        parallel to each other, it adds one to the other and removes the other one.

        Example: If we have two generators g1 and g2, and g1 = 2*g2,
        then we update g1 as g1 = g1 + g2 = 3*g2 and remove g2. 
        '''

        max_angle = 0.05 * math.pi / 180
        threshold = 1 - math.sin(max_angle)

        # Step 1: Stack Gc and Ac
        G = np.block([
            [hz.Gc],
            [hz.Ac]
        ])

        norms = np.linalg.norm(G, axis = 0)             # Compute norm of all columns
        zeros = np.where(norms == 0)[0]                 # Find all indices of 'norms' that are zero
        G = np.delete(arr = G, obj = zeros, axis = 1)   # Remove all 'zero' norm columngs from G
        ng = G.shape[1]

        i = 0; j = 0; k = 0
        while i < ng - k:
            g1_unit = G[:, i] / np.linalg.norm(G[:, i]) # Unit vector of g1

            j = i + 1
            while j < ng - k:

                g2 = G[:, j]
                g2_unit = g2 / np.linalg.norm(g2)       # Unit vector of g2

                if np.abs(np.dot(g1_unit.T, g2_unit)) >= threshold:
                    G[:, i] = G[:, i] + g2
                    G = np.delete(G, j, axis=1)         # Remove the second generator
                    k += 1
                else:
                    j += 1

            i +=1

        return HybridZonotope(G[:hz.dim, :], hz.Gb, hz.C, G[hz.dim:, :], hz.Ab, hz.b)    

    def redundant_c_hz(self, hz: HybridZonotope) -> HybridZonotope:
        '''
        Reduces the number of constraints of a hybrid Zonotope.

        In this method we are removing the following constraints:
            - Any constraint whose constraint matrix component (the particular row in Ac) is all zeros
            - Any constraint that there is another constraint that is equivalent to it
                e.g., x + y = 1 and 2x + 2y = 2, 5x + 5x = 5 only one out of these three constraints will be kept
        '''
        max_angle = 0.05 * math.pi / 180
        threshold = 1 - math.sin(max_angle)

        A = np.block([hz.Ac, hz.Ab, hz.b])

        norms = np.linalg.norm(A, axis = 1)             # Compute norm of all rows
        zeros = np.where(norms == 0)[0]                 # Find all indices of 'norms' that are zero
        A = np.delete(arr = A, obj = zeros, axis = 0)   # Remove all 'zero' norm columngs from G
        nc = A.shape[0]

        i = 0; j = 0; k = 0
        while i < nc - k:
            c1_unit = A[i, :] / np.linalg.norm(A[i, :]) # Unit vector of c1

            j = i + 1
            while j < nc - k:
                c2 = A[j, :]
                c2_unit = c2 / np.linalg.norm(c2)       # Unit vector of c2

                if np.abs(np.dot(c1_unit.T, c2_unit)) >= threshold:
                    A = np.delete(A, j, axis=0)         # Remove the second constraint
                    k += 1
                else:
                    j += 1

            i +=1


        return HybridZonotope(hz.Gc, hz.Gb, hz.C, A[:, :hz.Ac.shape[1]], A[:, hz.Ac.shape[1]:-1], A[:, -1].reshape((A.shape[0], 1)))    

    def find_E_hz_slow(self, hz):
        '''
        This method finds the E interval by solving 2*ng MILPs in equations (6.2a, 6.2b) from Section 6.1.2 in [7]
        This method provides the exact E bounds but it is generally more computationally expensive
        '''
        
        ## Step 1: Create a model
        model = gp.Model('intervals_cz')
        model.Params.OutputFlag = 0         # Disable verbose output

        ## Step 2: Create the variables
        x_c = model.addMVar(shape = (hz.ng, ), lb = np.array([-1] * hz.ng), ub = np.array([1] * hz.ng), vtype = np.array([gp.GRB.CONTINUOUS] * hz.ng), name = 'x_c')
        x_b = model.addMVar(shape = (hz.nb, ), lb = np.array([-1] * hz.nb), ub = np.array([1] * hz.nb), vtype = np.array([gp.GRB.INTEGER] * hz.nb), name = 'x_b')

        # Enforce that x_b only takes values in {-1, 1}^hz.nb
        for i in range(hz.nb):
            model.addConstr(x_b[i] * x_b[i] == 1 )

        # Compute the infinity norm of x_c
        norm_inf = model.addMVar(shape = 1, lb = 0, vtype = gp.GRB.CONTINUOUS, name = 'norm_inf')

        ## Step 3: Add constraints
        rhs = hz.b                          # Right hand side of equality constraint equation
        lhs = hz.Ac @ x_c + hz.Ab @ x_b     # Left hand side of equality constraint equation
        for left, right in zip(lhs, rhs):
            model.addConstr(left == right)
        
        model.addConstr(norm_inf == gp.norm(x_c, gp.GRB.INFINITY))  # Use the 'norm' General constraint helper function from the gurobi API

        x_L = []
        for g in range(hz.ng):
            model.setObjective(x_c[g], gp.GRB.MINIMIZE)
            model.optimize()
            x_L.append(x_c[g].X)

        x_U = []
        for g in range(hz.ng):
            model.setObjective(x_c[g], gp.GRB.MAXIMIZE)
            model.optimize()
            x_U.append(x_c[g].X)

        x_U = np.array(x_U); x_L = np.array(x_L)

        E = np.block([x_L.reshape(-1, 1), x_U.reshape(-1, 1)])

        return E

    def find_E_hz_fast(self, hz):
        '''
        This method computes the Intervals of the input hybrid zonotope according to Algorithm 1 in [6]
        which is then adapted to work for Hybrid zonotopes.
        This method does not provide the exact E bounds but it is generally more computationally efficient
        '''

        # Step 1: Initialize intervals Ej and Rj as Ej <- [-1, 1], Rj <- [-inf, inf], i,j <- 1
        E = np.array([ [-1, 1] for g in range(hz.ng) ])
        Eb = np.array([ [-1, 1] for b in range(hz.nb) ])
        R = np.array([ [-np.inf, np.inf] for g in range(hz.ng) ])

        A = hz.Ac
        epsilon = 1e-5
        iterations = 50  # Maximum number of iterations

        for iter in range(iterations):
            for i in range(hz.nc):
                for j in range(hz.ng):
                    if abs(A[i, j]) >= epsilon:
                        a_rc_inv = (1/hz.Ac[i, j])
                        R_rc = np.array([hz.b[i,0], hz.b[i,0]])

                        tempc = np.array([0.0, 0.0])
                        tempb = np.array([0.0, 0.0])
                        for k in range(hz.ng):
                            if k != j:
                                tempc = self.interval_add(tempc, self.interval_scalar_mul(hz.Ac[i, k], E[k, :]))
                        for b in range(hz.nb):
                            tempb = self.interval_add(tempb, self.interval_scalar_mul(hz.Ab[i, b], Eb[b, :]))
                        R_rc = self.interval_sub(R_rc, tempc)
                        R_rc = self.interval_sub(R_rc, tempb)
                        R_rc = self.interval_scalar_mul(a_rc_inv, R_rc)


                        R[j] = self.intesection_intervals(R[j], R_rc)
                        E[j] = self.intesection_intervals(E[j], R[j])
        return E

    def rref_hz(self, hz):
        '''
        This method computes the reduced row echelon form of matrix [A | b] for a linear system of equations (Ax = b)
        using Gauss-Jordan Elimination with full pivoting.

        - FUTURE IMPROVEMENTS: Experiment with other preconditioning strategies
        '''
        A = np.block([hz.Ac, hz.Ab, hz.b])

        rows = A.shape[0]

        pivots = []
        for r in range(rows):
            # Find the pivot row and column
            pivot, pivots = self.find_pivot(A[r, :-1], pivots)

            # Check if there is a new pivot
            if len(pivots) < r + 1:
                continue
            
            # Normalize the pivot row to turn the pivot element into 1
            A[r, :] = A[r, :] / A[r, pivot]

            # Turn the elements of all other rows in the pivot column to zero
            for r2 in range(rows):
                if r2 != r:
                    A[r2, :] = A[r2, :] - A[r2, pivot] * A[r, :]

        Ac = A[:, :hz.Ac.shape[1]]
        Ab = A[:, hz.Ac.shape[1]:-1]
        b = A[:, -1].reshape((A.shape[0], 1))

        return HybridZonotope(hz.Gc, hz.Gb, hz.C, Ac, Ab, b)

    def find_pivot(self, row, pivots):
        ''' 
        This method finds the pivot row and column for the Gauss-Jordan Elimination
        '''

        abs_row = np.abs(row)
        
        pivot = 0   # Init with dummy value
        
        for c in range(abs_row.shape[0]):
            if abs_row[c] != 0 and c not in pivots:
                pivot = c
                pivots.append(c)
                break

        return pivot, pivots
    
    def intervals_hz(self, hz):
        '''
        This method computes the Intervals of the input hybrid zonotope
        '''

        # Step 1: Initialize intervals Ej and Rj as Ej <- [-1, 1], Rj <- [-inf, inf], i,j <- 1
        E = np.array([ [-1, 1] for g in range(hz.ng + hz.nb) ])
        R = np.array([ [-np.inf, np.inf] for g in range(hz.ng + hz.nb) ])
        i = 0; j = 0

        A = np.block([hz.Ac, hz.Ab])

        while i < (hz.nc):
            while j < (hz.ng + hz.nb):
                if A[i, j] != 0:
                    a_ij_inv = (1/A[i, j])
                    sum = 0
                    for k in range(hz.ng + hz.nb):
                        if k != j:
                            sum += A[i, k] * E[k]
                    gen_val = a_ij_inv * hz.b[i,0] - a_ij_inv * sum
                    R[j] = self.intesection_intervals(R[j], gen_val)
                    E[j] = self.intesection_intervals(E[j], R[j])

                j += 1
            i += 1
            j = 0

        return E, R
    

    # ################################################3
    # Interval methods

    def interval_add(self, x, y):
        lb = min(x[0], x[1]); ub = max(x[0], x[1]); x[0] = lb; x[1] = ub
        lb = min(y[0], y[1]); ub = max(y[0], y[1]); y[0] = lb; y[1] = ub

        lb = x[0] + y[0]
        ub = x[1] + y[1]
        
        return np.array([lb, ub])

    def interval_sub(self, x, y):
        lb = min(x[0], x[1]); ub = max(x[0], x[1]); x[0] = lb; x[1] = ub
        lb = min(y[0], y[1]); ub = max(y[0], y[1]); y[0] = lb; y[1] = ub

        lb = x[0] - y[1]
        ub = x[1] - y[0]
        
        return np.array([lb, ub])

    def interval_mul(self, x, y):
        '''
        This method implements the multiplication between two interval sets
        '''
        lb = min(x[0], x[1]); ub = max(x[0], x[1]); x[0] = lb; x[1] = ub
        lb = min(y[0], y[1]); ub = max(y[0], y[1]); y[0] = lb; y[1] = ub

        xl = min(x[0], x[1]); xu = max(x[0], x[1])
        yl = min(y[0], y[1]); yu = max(y[0], y[1])

        if yl >= 0 and xl >= 0:
            lb = xl*yl
            ub = xu*yu
        elif yl >= 0 and (xl <= 0 and xu >= 0):
            lb = xl*yu
            ub = xu*yu
        elif yl >= 0 and xu <= 0:
            lb = xl*yu
            ub = xu*yl
        elif (yl <= 0 and yu >= 0) and xl >= 0:
            lb = xu*yl
            ub = xu*yu
        elif (yl <= 0 and yu >= 0) and (xl <= 0 and xu >= 0):
            lb = min(xl*yu, xu*yl)
            ub = max(xl*yl, xu*yu)
        elif (yl <= 0 and yu >= 0) and xu <= 0:
            lb = xl*yu
            ub = xl*yl
        elif yu <= 0 and xl >= 0:
            lb = xu*yl
            ub = xl*yu
        elif yu <= 0 and (xl <= 0 and xu >= 0):
            lb = xu*yl
            ub = xl*yl
        elif yu <= 0 and xu <= 0:
            lb = xu*yu
            ub = xl*yl


        return np.array([lb, ub])
            
    def interval_scalar_mul(self, k, x):
        '''
        This method implements the multiplication between a scalar and an interval set
        '''
        lb = min(x[0], x[1]); ub = max(x[0], x[1]); x[0] = lb; x[1] = ub

        if k >= 0:
            lb = k*x[0]
            ub = k*x[1]
        else:
            lb = k*x[1]
            ub = k*x[0]

        return np.array([lb, ub])        

    def infinum(self, S):
        '''
        This method computes all the infinum elements of an n-dimensional interval set
        '''
        inf = np.zeros((S.shape[0], 1))
        for j, e in enumerate(S):
            inf[j, 0] = min(S[j, 0], S[j, 1])

        return inf

    def supremum(self, S):
        '''
        This method computes all the supremum elements of an n-dimensional interval set
        '''
        sup = np.zeros((S.shape[0], 1))
        for j, e in enumerate(S):
            sup[j, 0] = max(S[j, 0], S[j, 1])

        return sup     

    def is_inside_interval(self, interval_1, interval_2):
        '''
        Check if interval_1 is a subset of interval_2
        '''

        # sort the intervals
        l = min(interval_1[0], interval_1[1])
        r = max(interval_1[0], interval_1[1])

        intersection = self.intesection_intervals(interval_1, interval_2)

        if intersection[0] == l and intersection[1] == r:
            return True
        else:
            return False

    def is_inside_1(self, interval):
        '''
        Check if interval_1 is a subset of interval_2
        '''
        epsilon = 1e-3
        # sort the intervals
        l = min(interval[0], interval[1])
        r = max(interval[0], interval[1])

        # if l >= -1 + epsilon and r <= 1 - epsilon:
        if l >= -1 and r <= 1:
            return True
        else:
            return False

    def intesection_intervals(self, interval_1, interval_2):
        l1 = min(interval_1[0], interval_1[1])
        r1 = max(interval_1[0], interval_1[1])        

        l2 = min(interval_2[0], interval_2[1])
        r2 = max(interval_2[0], interval_2[1])        

        if (l2 > r1 or r2 < l1):
            pass

        # Else update the intersection
        else:
            l1 = max(l1, l2)
            r1 = min(r1, r2)


        return np.array([l1, r1])


