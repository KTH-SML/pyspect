import numpy as np
from hz_reachability.sets import HybridZonotope
from hz_reachability.hz_impl import HZImpl

from hz_reachability.auxiliary_operations import ZonoOperations


class ParkingSpace:
    def __init__(self):
        self.lw = 0.1
        self.lh_v = 1.9         # Lane height of vertical lanes [m]
        self.lh_h = 2.8         # Lane height of horizontal lanes [m]
        self.zono_op = ZonoOperations()
        self.remove_redundant = False

    @property
    def state_space(self):
        # Vertical Road Sections (Left)
        Gc = np.diag(np.array([ self.lw/2  , self.lh_v/2]))
        Gb = np.array([ 
            [0.9, 0.45], 
            [0.0 , 0.0]
            ])
        c = np.array([ [0.0], [0.0]])
        Ac = np.zeros((0, 2))
        Ab = np.zeros((0, 2))
        b = np.zeros((0, 1))

        road_v = HybridZonotope(Gc, Gb, c, Ac, Ab, b)

        # Horizontal Road Sections (Exterior)
        Gc = np.diag(np.array([ self.lh_h/2  , self.lw/2]))
        Gb = np.array([ [0.0], [0.9]])
        c = np.array([ [0.0], [0.0]])
        Ac = np.zeros((0, 2))
        Ab = np.zeros((0, 1))
        b = np.zeros((0, 1))

        road_h_ext = HybridZonotope(Gc, Gb, c, Ac, Ab, b)

        # # Horizontal Road Sections (Middle)
        Gc = np.diag(np.array([ self.lh_h/2 + 0.2  , self.lw/2]))
        Gb = np.zeros((2, 0))
        c = np.array([ [0.2], [0.0]])
        Ac = np.zeros((0, 2))
        Ab = np.zeros((0, 0))
        b = np.zeros((0, 1))

        road_h_mid = HybridZonotope(Gc, Gb, c, Ac, Ab, b)

        road_h = self.zono_op.union(road_h_ext, road_h_mid)
        if self.remove_redundant:
            road_h = self.zono_op.redundant_c_gc_hz_v2(road_h)
            road_h = self.zono_op.redundant_c_gc_hz_v1(road_h)

        road = self.zono_op.union(road_v, road_h)
        if self.remove_redundant:
            road = self.zono_op.redundant_c_gc_hz_v2(road)
            road = self.zono_op.redundant_c_gc_hz_v1(road)

        return road

    @property
    def input_space(self):
        # Maximum rate of change in velocity (acceleration)
        ng = 2; nc = 0; nb = 0

        Gc = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        c = np.array([ [0.0], [0.0] ]); Gb = np.zeros((ng, nb))
        Ac = np.zeros((nc, ng)); Ab = np.zeros((nc, nb)); b = np.zeros((nc, 1))

        return HybridZonotope(Gc, Gb, c, Ac, Ab, b) 

    @property
    def augmented_state_space(self):
        # Vertical Road Sections (Left)
        Gc = np.diag(np.array([ self.lw/2  , self.lh_v/2, 0.0, 1e-4 ]))
        # Gc = np.diag(np.array([ self.lw/2  , self.lh_v/2, 1.0, 1e-4 ]))
        Gb = np.array([ [0.45], [0.0], [0.0], [-1.0] ])
        c = np.array([ [-0.9], [0.0], [ 0.0], [ 0.0] ])
        Ac = np.zeros((0, 4))
        Ab = np.zeros((0, 1))
        b = np.zeros((0, 1))

        road_v_left = HybridZonotope(Gc, Gb, c, Ac, Ab, b)

        # Vertical Road Sections (Right)
        Gc = np.diag(np.array([ self.lw/2  , self.lh_v/2, 0.0, 1e-4 ]))
        # Gc = np.diag(np.array([ self.lw/2  , self.lh_v/2, 1.0, 1e-4 ]))
        Gb = np.array([ [0.45], [0.0], [0.0], [-1.0] ])
        c = np.array([ [0.9], [0.0], [ 0.0], [ 0.0] ])
        Ac = np.zeros((0, 4))
        Ab = np.zeros((0, 1))
        b = np.zeros((0, 1))

        road_v_right = HybridZonotope(Gc, Gb, c, Ac, Ab, b)

        # Horizontal Road Sections (Exterior)
        Gc = np.diag(np.array([ self.lh_h/2  , self.lw/2, 1e-4, 0.0 ]))
        # Gc = np.diag(np.array([ self.lh_h/2  , self.lw/2, 1e-4, 1.0 ]))
        Gb = np.array([ [0.0], [0.9], [1.0], [0.0] ])
        c = np.array([ [0.0], [0.0], [ 0.0], [ 0.0] ])
        Ac = np.zeros((0, 4))
        Ab = np.zeros((0, 1))
        b = np.zeros((0, 1))

        road_h_ext = HybridZonotope(Gc, Gb, c, Ac, Ab, b)

        # # Horizontal Road Sections (Middle)
        Gc = np.diag(np.array([ self.lh_h/2 + 0.2  , self.lw/2, 1e-4, 0.0 ]))
        # Gc = np.diag(np.array([ self.lh_h/2 + 0.2  , self.lw/2, 1e-4, 1.0 ]))
        Gb = np.zeros((4, 0))
        c = np.array([ [0.2], [0.0], [ 1.0], [ 0.0] ])
        Ac = np.zeros((0, 4))
        Ab = np.zeros((0, 0))
        b = np.zeros((0, 1))

        road_h_mid = HybridZonotope(Gc, Gb, c, Ac, Ab, b)

        road_v = self.zono_op.union(road_v_left, road_v_right)
        if self.remove_redundant:
            road_v = self.zono_op.redundant_c_gc_hz_v2(road_v)
            road_v = self.zono_op.redundant_c_gc_hz_v1(road_v)

        road_h = self.zono_op.union(road_h_ext, road_h_mid)
        if self.remove_redundant:
            road_h = self.zono_op.redundant_c_gc_hz_v2(road_h)
            road_h = self.zono_op.redundant_c_gc_hz_v1(road_h)

        road = self.zono_op.union(road_v, road_h)
        if self.remove_redundant:
            road = self.zono_op.redundant_c_gc_hz_v2(road)
            road = self.zono_op.redundant_c_gc_hz_v1(road)

        return road

class EmptySpace:
    
    def __init__(self, min_bounds, max_bounds):
        assert len(min_bounds) == len(max_bounds)
        self.N = len(min_bounds)
        self.max_bounds = np.array(max_bounds)
        self.min_bounds = np.array(min_bounds)
        self.zono_op = ZonoOperations()
        self.remove_redundant = False

    @property
    def state_space(self):
        ng = self.N; nc = 0; nb = 0
        
        Gc = np.diag(self.max_bounds - self.min_bounds) / 2
        c = np.array(self.max_bounds + self.min_bounds).reshape(-1, 1) / 2

        Gb = np.zeros((ng, nb))
        Ac = np.zeros((nc, ng))
        Ab = np.zeros((nc, nb))
        b = np.zeros((nc, 1))

        return HybridZonotope(Gc=Gc, Gb=Gb, C=c, Ac=Ac, Ab=Ab, b=b)
