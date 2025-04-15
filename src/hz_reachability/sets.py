import numpy as np

'''
The properties of the hybrid zonotope class follow the work in [1]

References:

[1] T. Bird Hybrid Zonotopes: a new set representation for reachability analyisis of mixed logical dynamical systems
'''

class HybridZonotope:
    def __init__(self, Gc: np.ndarray, Gb: np.ndarray, C: np.ndarray, Ac: np.ndarray, Ab:np.ndarray, b: np.ndarray) -> None:
        """
        Description
        -----------
        Hybrid Zonotope HZ = (Gc, Gb, C, Ac, Ab, b)

        - Gc: Generators of the continuous component
            - [g_1, g_2, ..., g_ng] where g_1, g_2, ..., g_ng are column vectors of dimension n
        - Gb: Generators of the binary component
            - [g_1, g_2, ..., g_nb] where g_1, g_2, ..., g_nb are column vectors of dimension n
        - C : Center of the constrained zonotope
            - [c1, c2, ..., cn]^T where c1, c2, ..., cn are scalars
        - Ac: matrix of linear constraints of dimensions (nc, ng), nc is the number of constraints
        - Ab: Binary, dimensions (nc, nb)
        - b : vector of linear constraints of dimensions (nc, 1), nc is the number of constraints
        """

        assert C.shape[0] == Gc.shape[0] == Gb.shape[0], 'c, Gc and Gb must match in state space'
        assert Gc.shape[1] == Ac.shape[1], 'Number of columns in Ac must match number of continuous generators'
        assert Gb.shape[1] == Ab.shape[1], 'Number of columns in Ab must match number of binary generators'
        assert Ac.shape[0] == Ab.shape[0] == b.shape[0], 'Number of constraints must be consinstent across Ac, Ab, and b'
        
        self.C = C.reshape(-1, 1)   # Reshape center
        self.Gc = Gc
        self.Gb = Gb
        self.Ac = Ac
        self.Ab = Ab
        self.b = b

    def astuple(self) -> tuple:
        return (self.C, self.Gc, self.Gb, self.Ac, self.Ab, self.b)

    @property
    def dim(self) -> int:
        return self.C.shape[0]
    
    @property
    def ng(self) -> int:
        '''
        Number of generators
        '''
        return self.Gc.shape[1]
    
    @property
    def nb(self) -> int:
        '''
        Number of binary generators
        '''
        return self.Gb.shape[1]

    @property
    def nc(self) -> int:
        '''
        Continuous constraints
        '''
        return self.Ac.shape[0]
    
    @property
    def od(self) -> int:
        '''
        Order of continuous component
        '''
        return (self.ng - self.nb) / self.dim
    
    @property
    def ob(self) -> int:
        '''
        TODO:
        Order of binary component

        ob = nb / | log_{2}(T) |
        '''
        pass


class ConstrainedZonotope:
    def __init__(self, G: np.ndarray, C: np.ndarray, A: np.ndarray, b: np.ndarray) -> None:
        """
        Constrained Zonotope CZ = (G, C, A, b)

        - G: generators of the constrained zonotope
            - [g_1, g_2, ..., g_ng] where g_1, g_2, ..., g_ng are column vectors of dimension n        
        - C: center of the constrained zonotope
            - [c1, c2, ..., cn]^T where c1, c2, ..., cn are scalars
        - A: matrix of linear constraints of dimensions (nc, ng), nc is the number of constraints
        - b: vector of linear constraints of dimensions (nc, 1), nc is the number of constraints
        """
        self.C = C.reshape(-1, 1)   # Reshape center
        self.G = G
        self.A = A
        self.b = b

    @property
    def dim(self) -> int:
        return self.C.shape[0]
    
    @property
    def ng(self) -> int:
        """
        Number of generators
        """
        return self.G.shape[1]
    
    @property
    def nc(self) -> int:
        """
        Number of constraints
        """
        return self.A.shape[0]
    
    @property
    def order(self) -> int:
        return (self.ng - self.nc)/self.dim
        


import numpy as np
import cdd as pcdd # Enumerate vertices of a polytope
from scipy.spatial import ConvexHull

from scipy.spatial import HalfspaceIntersection

from scipy.optimize import linprog


import matplotlib.pyplot as plt

'''
The properties of the constrained zonotope class follow the Scott's, et al. work
'Constrained zonotopes: A new tool for set-based estimation and fault detection'

References:

[1] Liren Yang, et al. 'Efficient Backward Reachability Using the Minkowski Difference of Constrained Zonotopes'
# [?] Scott, et al. 'Constrained zonotopes: A new tool for set-based estimation and fault detection'
'''



class ConstrainedZonotope:
    def __init__(self, G: np.ndarray, C: np.ndarray, A: np.ndarray, b: np.ndarray) -> None:
        '''
        Constrained Zonotope CZ = (G, C, A, b)

        - G: generators of the constrained zonotope
            - [g_1, g_2, ..., g_ng] where g_1, g_2, ..., g_ng are column vectors of dimension n        
        - C: center of the constrained zonotope
            - [c1, c2, ..., cn]^T where c1, c2, ..., cn are scalars
        - A: matrix of linear constraints of dimensions (nc, ng), nc is the number of constraints
        - b: vector of linear constraints of dimensions (nc, 1), nc is the number of constraints
        '''
        self.C = C.reshape(-1, 1)   # Reshape center
        self.G = G
        self.A = A
        self.b = b

    @property
    def dim(self) -> int:
        return self.C.shape[0]
    
    @property
    def ng(self) -> int:
        '''
        Number of generators
        '''
        return self.G.shape[1]
    
    @property
    def nc(self) -> int:
        '''
        Number of constraints
        '''
        return self.A.shape[0]
    
    @property
    def order(self) -> int:
        return (self.ng - self.nc)/self.dim
        
    def g2v(self):
        '''
        This method returns the vertices of the constrained zonotope

        See work in [1] for details
        '''
        Ap = np.concatenate([
            self.A,
            -self.A,
            np.eye(self.ng),
            -np.eye(self.ng)
        ])
        bp = np.concatenate([
            self.b,
            -self.b,
            np.ones((2*self.ng, 1))
        ])

        vertices, rays, empty_flag = self.h2v(Ap, bp)
        # vertices, rays, empty_flag = self.h2v_v2(Ap, bp)

        if empty_flag == True:
            return np.zeros((0, self.dim)), np.zeros((0, self.dim)), empty_flag
        
        # Apply the affine transformation
        vertices = self.C + self.G @ vertices.T
        vertices = vertices.T


        if rays.size == 0:
            rays = self.G @ rays.T
            rays = rays.T

        vertices = np.array(vertices)

        # Loop through all vertices to delete any duplicates
        i = 0
        while i < vertices.shape[0]:
            j = i + 1
            while j < vertices.shape[0]:
                if np.allclose(vertices[i], vertices[j]):
                    vertices = np.delete(vertices, j, axis=0)
                else:
                    j += 1
            i += 1

        if vertices.shape[0] < 3:
            return np.zeros((0, self.dim)), np.zeros((0, self.dim)), True

        # Remove vertices not on the edge of the convex hull
        try:
            hull = ConvexHull(vertices)
            vertices = vertices[hull.vertices]
        except:
            return np.zeros((0, self.dim)), np.zeros((0, self.dim)), True

        return vertices, rays, False



    def remove_colinear(self, vertices: np.ndarray):
        '''
        This method removes colinear vertices

        For each triplet of vertices, if the angle between the vectors formed by the triplet is 0 or pi, then the middle vertex is removed
        '''
        # Loop through all triplets of vertices
        i = 0
        while i < vertices.shape[0]:
            j = i + 1
            while j < vertices.shape[0]:
                k = j + 1
                while k < vertices.shape[0]:
                    if self.are_colinear(vertices[i], vertices[j], vertices[k]):
                        # Find the middle vertex among i, j, and k
                        dist_ij = np.linalg.norm(vertices[i] - vertices[j])
                        dist_ik = np.linalg.norm(vertices[i] - vertices[k])
                        dist_jk = np.linalg.norm(vertices[j] - vertices[k])
                        
                        if dist_ik > dist_ij and dist_jk > dist_ik:
                            vertices = np.delete(vertices, i, axis=0)
                        elif dist_ik > dist_ij and dist_jk < dist_ik:
                            vertices = np.delete(vertices, j, axis=0)
                        else:
                            vertices = np.delete(vertices, k, axis=0)

                    else:
                        k += 1
                j += 1
            i += 1


        return vertices

    def are_colinear(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray):
        '''
        This method checks if three n-dimensional vertices are on the same line segment
        '''
        # Compute the vector between v1 and v2
        v12 = v2 - v1
        # Compute the vector between v1 and v3
        v13 = v3 - v1

        # Compute the angle between v12 and v13
        angle = np.arccos(np.dot(v12, v13)/(np.linalg.norm(v12)*np.linalg.norm(v13)))

        # If the angle is 0 or pi, then the vertices are colinear
        if np.isclose(angle, 0) or np.isclose(angle, np.pi):
            return True
        else:
            return False
        
        # # Check if the angle between the vectors formed by the triplet is 0 +- threshold or pi +- threshold
        # threshold = 1e-3
        # if np.allclose(np.dot(v1 - v2, v3 - v2), 0, atol=threshold) or np.allclose(np.dot(v1 - v2, v3 - v2), -1, atol=threshold):
        #     return True
        # else:
        #     return False        
        
    def h2v(self, A, b):
        '''
        Given a Polytope defined by Ax <= b
        Return the vertices and rays of the Polytope
        '''
        M = np.hstack( (b, -A) )

        # Convert to cdd format
        P = pcdd.Matrix(M)
        P.rep_type=pcdd.RepType.INEQUALITY
        P = pcdd.Polyhedron(P)
        Q = P.get_generators()
        # Convert to numpy array
        Q = np.array(Q)

        # Check if Q is empty
        if Q.size == 0:
            vertices = None
            rays = None
            empty_flag = True
        else:
            # Extract vertices and rays
            vertices = Q[Q[:, 0] == 1, 1:]
            rays = Q[Q[:, 0] == 0, 1:]
            empty_flag = False

        return vertices, rays, empty_flag 
    
