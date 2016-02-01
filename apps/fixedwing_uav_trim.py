import numpy as np
import mpl_toolkits.mplot3d as a3
import pylab as pl
from uav.fixed_wing import FixedWingUAV#, FixedWingUAVDynamics
from viewer.viewer import Viewer

class UAVViewer(Viewer):
    fuse_l1 = 5.
    fuse_l2 = 2.5
    fuse_l3 = 10.
    fuse_h = 3.
    fuse_w = 3.
    wing_w = 15.
    wing_l = 3.
    tail_h = 4.
    tailwing_w = 7.5
    tailwing_l = 1.5
    def __init__(self, ax, x0, R):
        self.vertices = np.matrix(np.zeros((16, 3), dtype = np.double))
        self.vertices[0, :] =  [self.fuse_l1, 0.0, 0.0]
        self.vertices[1, :] = [self.fuse_l2, 0.5*self.fuse_w, 0.5*self.fuse_h]
        self.vertices[2, :] = [self.fuse_l2, -0.5*self.fuse_w, 0.5*self.fuse_h]
        self.vertices[3, :] = [self.fuse_l2, -0.5*self.fuse_w, -0.5*self.fuse_h]
        self.vertices[4, :] = [self.fuse_l2, 0.5*self.fuse_w, -0.5*self.fuse_h]
        self.vertices[5, :] = [-self.fuse_l3, 0., 0.]
        self.vertices[6, :] = [0, 0.5 * self.wing_w, 0.0]
        self.vertices[7, :] = [-self.wing_l, 0.5 * self.wing_w, 0.0]
        self.vertices[8, :] = [-self.wing_l, -0.5 * self.wing_w, 0.0]
        self.vertices[9, :] = [0, -0.5 * self.wing_w, 0.0]
        self.vertices[10, :] = [-(self.fuse_l3 - self.tailwing_l), 0.5 * self.tailwing_w, 0.]
        self.vertices[11, :] = [-self.fuse_l3, 0.5 * self.tailwing_w, 0.]
        self.vertices[12, :] = [-self.fuse_l3, -0.5 * self.tailwing_w, 0.]
        self.vertices[13, :] = [-(self.fuse_l3 - self.tailwing_l), -0.5 * self.tailwing_w, 0.]
        self.vertices[14, :] = [-(self.fuse_l3 - self.tailwing_l), 0.0, 0.]
        self.vertices[15, :] = [-self.fuse_l3, 0.0, -self.tail_h]
        self.vertices = 0.3048 * self.vertices
        self.vertices = self.vertices * R + x0[0:3]
        self.nose = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 1, 4]]
        self.fuselage = [[5, 3, 2], [5, 2, 1], [5, 3, 4], [5, 4, 1]]
        self.wing = [[6, 7, 8, 9]]
        self.tail_wing = [[10, 11, 12, 13]]
        self.tail = [[5, 14, 15]]
        self.faces = [self.nose, self.fuselage, self.wing, self.tail_wing, self.tail]
        super(UAVViewer, self).__init__(ax, self.vertices, self.faces, ['r', 'g', 'g', 'g', 'y'])
        
class AppFixedWingUAVTrim(FixedWingUAV):
    def __init__(self, x0, t0, config, ax):
        super(AppFixedWingUAVTrim, self).__init__(x0, t0, config)
        self.viewer = UAVViewer(ax, x0, self.R_bv(x0[6:9]))
        
    def update_view(self):
        new_vertices = self.viewer.rotate(self.R_bv(self.dynamics.x[6:9])) + self.dynamics.x[0:3]
        self.viewer.update(new_vertices)
    
    def trim(self, Va, gamma, radius, max_iters):
        trimmed_state, trimmed_control_inputs = self.dynamics.trim(Va, gamma, radius, epsilon=1e-8, kappa=1e-6, max_iters=max_iters)
        #self.set_state(trimmed_state, 0.)
        self.set_control_inputs(trimmed_control_inputs)