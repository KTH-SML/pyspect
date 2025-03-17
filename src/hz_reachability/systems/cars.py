import numpy as np
from hz_reachability.sets import HybridZonotope
from hz_reachability.hz_impl import HZImpl

step_size = 0.05

class CarLinearModel2D:
    '''
    This class defines a discrete-time linear dynamics model without disturbances.

    x_{k+1} = Ax_k + Bu_k
    
    These dynamics are used for the propagation of the backward reachable set

    '''
    def __init__(self) -> None:
        self.vx_max = 1.0   # m/s Maximum permissible velocity in x-axis
        self.vy_max = 1.0   # m/s Maximum permissible velocity in y-axis

        # self.dt = 0.1       # [s] Time step (10 Hz)
        self.dt = step_size       # [s] Time step (20 Hz)
        
        self.A = np.array([
            [1.0, 0.0],     # x - position
            [0.0, 1.0],     # y - position
        ])
        self.B = np.array([
            [self.vx_max*self.dt,          0.0       ],     # x - velocity
            [       0.0         , self.vy_max*self.dt],     # y - velocity
        ])

        self.AB = np.hstack((self.A, self.B))

        self.input_spacce = self.get_input_space

    def get_input_space(self):
        """
        Description
        ------------
        """
        # Maximum rate of change in velocity (acceleration)
        ng = 2; nc = 0; nb = 0

        Gc = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        c = np.array([ [0.0], [0.0] ]); Gb = np.zeros((ng, nb))
        Ac = np.zeros((nc, ng)); Ab = np.zeros((nc, nb)); b = np.zeros((nc, 1))

        return HybridZonotope(Gc, Gb, c, Ac, Ab, b) 


class CarLinearModel4D:
    '''
    This class defines a discrete-time linear dynamics model without disturbances.

    x_{k+1} = Ax_k + Bu_k

    These dynamics are used for the propagation of the forward reachable set
    
    '''
    def __init__(self) -> None:
        a_max = 4.5 # [m/s^2]
        
        # self.dt = 0.1       # [s] Time step (10 Hz)
        self.dt = step_size       # [s] Time step (20 Hz)

        self.A = np.array([
            [1.0, 0.0, self.dt, 0.0    ],     # x - position
            [0.0, 1.0, 0.0    , self.dt],     # y - position
            [0.0, 0.0, 1.0    , 0.0    ],     # x - velocity
            [0.0, 0.0, 0.0    , 1.0    ],     # y - velocity
        ])
        self.B = np.array([
            [       0.0         ,     0.0      ],     # x - position
            [       0.0         ,     0.0      ],     # y - position
            [   a_max*self.dt   ,     0.0      ],     # x - velocity
            [       0.0         , a_max*self.dt],     # y - velocity
        ])

        self.AB = np.hstack((self.A, self.B))
        self.input_spacce = self.get_input_space

    def get_input_space(self):
        """
        Description
        ------------
        """
        # Maximum rate of change in velocity (acceleration)
        ng = 2; nc = 0; nb = 0

        Gc = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        c = np.array([ [0.0], [0.0] ]); Gb = np.zeros((ng, nb))
        Ac = np.zeros((nc, ng)); Ab = np.zeros((nc, nb)); b = np.zeros((nc, 1))

        return HybridZonotope(Gc, Gb, c, Ac, Ab, b) 

class CircularBicycle5DLinearized:
    '''
    This class defines a discrete-time linear dynamics model without disturbances.

    x_{k+1} = Ax_k + Bu_k

    '''

    dt = step_size

    def __init__(self, z0=..., u0=..., step_size=...) -> None:
        if z0 is Ellipsis: z0 = [0] * 5
        if u0 is Ellipsis: u0 = [0] * 2
        if step_size is not Ellipsis: self.dt = step_size

        ## Continuous time linearization

        r, phi, v_r, v_phi, w = z0
        a, d = u0
        L = 0.32

        Ac = [[0.0, 0.0, 1.0,          0.0,   0.0],
              [0.0, 0.0, 0.0,          1.0,   0.0],
              [0.0, 0.0, 0.0,            w, v_phi],
              [0.0, 0.0,  -w,          0.0,  -v_r],
              [0.0, 0.0, 0.0,          d/L,   0.0]]

        Bc = [[0.0,                    0.0],
              [0.0,                    0.0],
              [0.0,                    0.0],
              [1.0,                    0.0],
              [0.0,                v_phi/L]]

        ## Discrete time linearization

        self.A = np.identity(5) + np.array(Ac)*self.dt
        self.B = np.array(Bc)*self.dt

        self.AB = np.hstack((self.A, self.B))
        self.input_spacce = self.get_input_space

    def get_input_space(self):
        """
        Description
        ------------
        """
        # Maximum rate of change in velocity (acceleration)
        ng = 2; nc = 0; nb = 0

        Gc = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        c = np.array([ [0.0], [0.0] ]); Gb = np.zeros((ng, nb))
        Ac = np.zeros((nc, ng)); Ab = np.zeros((nc, nb)); b = np.zeros((nc, 1))

        return HybridZonotope(Gc, Gb, c, Ac, Ab, b) 











