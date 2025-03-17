'''
This script provides zonotopic templates for different standard shapes.
'''
import numpy as np
from hz_reachability.sets import HybridZonotope


class HZShapes:

    nz = 5
    ng = 2
    nc = 0
    nb = 0

    def center(self):

        ## State Expression ##

        C = np.zeros((self.nz, 1))

        Gc = np.zeros((self.nz, self.ng))
        Gc[:2, :2] = np.array([
            [0.1, 0.0],
            [0.0, 0.1]
        ])

        Gb = np.zeros((self.nz, self.nb))

        ## Constraints Expression ##

        Ac = np.zeros((self.nc, self.ng))
        
        Ab = np.zeros((self.nc, self.nb))

        b = np.zeros((self.nc, 1))

        return HybridZonotope(Gc = Gc, Gb = Gb, C = C, Ac = Ac, Ab = Ab, b = b)
    
    def road_west(self):

        ## State Expression ##

        C = np.zeros((self.nz, 1))
        C[:2] = np.array([
            [-0.55],
            [0.0]
        ])

        Gc = np.zeros((self.nz, self.ng))
        Gc[:2, :2] = np.array([
            [0.45, 0.00],
            [0.00, 0.1]
        ])

        Gb = np.zeros((self.nz, self.nb))

        ## Constraints Expression ##

        Ac = np.zeros((self.nc, self.ng))

        Ab = np.zeros((self.nc, self.nb))

        b = np.zeros((self.nc, 1))

        return HybridZonotope(Gc = Gc, Gb = Gb, C = C, Ac = Ac, Ab = Ab, b = b)

    def road_east(self):

        ## State Expression ##

        C = np.zeros((self.nz, 1))
        C[:2] = np.array([
            [0.55],
            [0.0]
        ])

        Gc = np.zeros((self.nz, self.ng))
        Gc[:2, :2] = np.array([
            [0.45, 0.00],
            [0.00, 0.1]
        ])

        Gb = np.zeros((self.nz, self.nb))

        ## Constraints Expression ##

        Ac = np.zeros((self.nc, self.ng))

        Ab = np.zeros((self.nc, self.nb))

        b = np.zeros((self.nc, 1))

        return HybridZonotope(Gc = Gc, Gb = Gb, C = C, Ac = Ac, Ab = Ab, b = b)

    def road_north(self):

        ## State Expression ##

        C = np.zeros((self.nz, 1))
        C[:2] = np.array([
            [0.0],
            [0.55]
        ])

        Gc = np.zeros((self.nz, self.ng))
        Gc[:2, :2] = np.array([
            [0.1, 0.00],
            [0.00, 0.45]
        ])

        Gb = np.zeros((self.nz, self.nb))

        ## Constraints Expression ##

        Ac = np.zeros((self.nc, self.ng))

        Ab = np.zeros((self.nc, self.nb))

        b = np.zeros((self.nc, 1))

        return HybridZonotope(Gc = Gc, Gb = Gb, C = C, Ac = Ac, Ab = Ab, b = b)

    def road_south(self):

        ## State Expression ##

        C = np.zeros((self.nz, 1))
        C[:2] = np.array([
            [0.0],
            [-0.55]
        ])

        Gc = np.zeros((self.nz, self.ng))
        Gc[:2, :2] = np.array([
            [0.1, 0.00],
            [0.00, 0.45]
        ])

        Gb = np.zeros((self.nz, self.nb))

        ## Constraints Expression ##

        Ac = np.zeros((self.nc, self.ng))

        Ab = np.zeros((self.nc, self.nb))

        b = np.zeros((self.nc, 1))

        return HybridZonotope(Gc = Gc, Gb = Gb, C = C, Ac = Ac, Ab = Ab, b = b)
