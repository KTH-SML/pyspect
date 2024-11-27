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
            - [g_1, g_2, ..., g_nc] where g_1, g_2, ..., g_nc are column vectors of dimension n
        - Gb: Generators of the binary component
            - [g_1, g_2, ..., g_nb] where g_1, g_2, ..., g_nb are column vectors of dimension n
        - C : Center of the constrained zonotope
            - [c1, c2, ..., cn]^T where c1, c2, ..., cn are scalars
        - Ac: matrix of linear constraints of dimensions (nc, ng), nc is the number of constraints
        - Ab: Binary, dimensions (nc, nb)
        - b : vector of linear constraints of dimensions (nc, 1), nc is the number of constraints
        """

        assert C.shape[0] == Gc.shape[0] == Gb.shape[0], 'Center and generators must have the same dimension'
        assert Gc.shape[1] == Ac.shape[1], 'Number of columns in Ac must match number of continuous generators'
        assert Gb.shape[1] == Ab.shape[1], 'Number of columns in Ab must match number of binary generators'
        assert Ac.shape[0] == Ab.shape[0] == b.shape[0], 'Number of constraints must be consinstent across Ac, Ab, and b'

        
        self.C = C.reshape(-1, 1)   # Reshape center
        self.Gc = Gc
        self.Gb = Gb
        self.Ac = Ac
        self.Ab = Ab
        self.b = b

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
        



