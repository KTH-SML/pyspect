'''
This script provides zonotopic templates for different standard shapes.
'''
import numpy as np
from hz_reachability.sets import HybridZonotope


class HZShapes:

    def center(self):
        """
        Description
        ------------

        Parameters
        -----------

        Returns
        -----------      
        """
        ng = 2; nc = 0; nb = 0
        # Continuous components
        C = np.zeros((2, 1))
        Gc = np.array([
            [0.1, 0.0],
            [0.0, 0.1]
        ])
        Ac = np.zeros((nc, 2))
        
        # Binary components
        b = np.zeros((nc, 1))
        Gb = np.zeros((ng, nb))
        Ab = np.zeros((nc, nb))

        return HybridZonotope(Gc = Gc, Gb = Gb, C = C, Ac = Ac, Ab = Ab, b = b)
    
    def road_west(self):
        ng = 2; nc = 0; nb = 0
        # Continuous components
        C = np.array([
            [-0.55],
            [0.0]
        ])
        Gc = np.array([
            [0.45, 0.00],
            [0.00, 0.1]
        ])
        Ac = np.zeros((nc, 2))

        # Binary components
        b = np.zeros((nc, 1))
        Gb = np.zeros((ng, nb))
        Ab = np.zeros((nc, nb))

        return HybridZonotope(Gc = Gc, Gb = Gb, C = C, Ac = Ac, Ab = Ab, b = b)

    def road_east(self):
        ng = 2; nc = 0; nb = 0
        # Continuous components
        C = np.array([
            [0.55],
            [0.0]
        ])
        Gc = np.array([
            [0.45, 0.00],
            [0.00, 0.1]
        ])
        Ac = np.zeros((nc, 2))

        # Binary components
        b = np.zeros((nc, 1))
        Gb = np.zeros((ng, nb))
        Ab = np.zeros((nc, nb))

        return HybridZonotope(Gc = Gc, Gb = Gb, C = C, Ac = Ac, Ab = Ab, b = b)

    def road_north(self):
        ng = 2; nc = 0; nb = 0
        # Continuous components
        C = np.array([
            [0.0],
            [0.55]
        ])
        Gc = np.array([
            [0.1, 0.00],
            [0.00, 0.45]
        ])
        Ac = np.zeros((nc, 2))

        # Binary components
        b = np.zeros((nc, 1))
        Gb = np.zeros((ng, nb))
        Ab = np.zeros((nc, nb))

        return HybridZonotope(Gc = Gc, Gb = Gb, C = C, Ac = Ac, Ab = Ab, b = b)

    def road_south(self):
        ng = 2; nc = 0; nb = 0
        # Continuous components
        C = np.array([
            [0.0],
            [-0.55]
        ])
        Gc = np.array([
            [0.1, 0.00],
            [0.00, 0.45]
        ])
        Ac = np.zeros((nc, 2))

        # Binary components
        b = np.zeros((nc, 1))
        Gb = np.zeros((ng, nb))
        Ab = np.zeros((nc, nb))

        return HybridZonotope(Gc = Gc, Gb = Gb, C = C, Ac = Ac, Ab = Ab, b = b)
