import numpy as np
from hz_reachability.sets import HybridZonotope

dt = 0.05

class DoubleIntegrator:

    def __init__(self, max_accel, dt=dt) -> None:
        
        ## Dynamics ##

        self.A = np.array([
            [1.0,  dt],
            [0.0, 1.0],
        ])
        self.B = np.array([
            [dt**2/2],
            [dt     ],
            # [0],
            # [1],
        ]) * max_accel

        self.AB = np.hstack((self.A, self.B))

        ## Input Space ##

        # Maximum rate of change in velocity (acceleration)
        ng = 1; nc = 0; nb = 0

        Gc = np.array([
            [1.0],
        ])

        c  = np.array([[0.0]])
        Gb = np.zeros((ng, nb))
        Ac = np.zeros((nc, ng))
        Ab = np.zeros((nc, nb))
        b  = np.zeros((nc, 1))

        self.input_space = HybridZonotope(Gc, Gb, c, Ac, Ab, b) 
