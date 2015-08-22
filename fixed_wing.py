import numpy as np
from viewer.viewer import UAVViewer
from dynamics import FixedWingUAVDynamics
#import scipy as sp
#import matplotlib.colors as colors    
class FixedWingUAV:
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
    
    def __init__(self, x0, t0, dynamics_config_file, ax):
        self.vertices = np.matrix(np.zeros((16, 3), dtype = np.double))
#        self.x = np.zeros((12,), dtype = np.double)
#        self.x = x0
#        self.t = t0
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
        
        self.nose = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 1, 4]]
        self.fuselage = [[5, 3, 2], [5, 2, 1], [5, 3, 4], [5, 4, 1]]
        self.wing = [[6, 7, 8, 9]]
        self.tail_wing = [[10, 11, 12, 13]]
        self.tail = [[5, 14, 15]]
        self.faces = [self.nose, self.fuselage, self.wing, self.tail_wing, self.tail]
        self.viewer = UAVViewer(ax, (self.rotate(x0[6:9]) + x0[0:3]), self.faces, ['r', 'g', 'g', 'g', 'y'])
        self.dynamics = FixedWingUAVDynamics(x0, t0, 0.001, dynamics_config_file)
    
    def move_to(self, pos):
        self.x[0:3] = pos
    
    def R_v1v(self, yaw_rad):
        cy = np.cos(yaw_rad)
        sy = np.sin(yaw_rad)
        R = np.matrix(([cy, sy, 0], [-sy, cy, 0], [0, 0, 1]), dtype = np.double)
        return R
    
    def R_v2v1(self, pitch_rad):
        cp = np.cos(pitch_rad)
        sp = np.sin(pitch_rad)
        R = np.matrix(([cp, 0, -sp], [0, 1, 0], [sp, 0, cp]), dtype = np.double)
        return R
        
    def R_bv2(self, roll_rad):
        cr = np.cos(roll_rad)
        sr = np.sin(roll_rad)
        R = np.matrix(([1, 0, 0], [0, cr, sr], [0, -sr, cr]), dtype = np.double)
        return R
    
    def R_bv(self, rpy_rad):
        yaw = rpy_rad[2]
        pitch = rpy_rad[1]
        roll = rpy_rad[0]
        R = self.R_bv2(roll) * self.R_v2v1(pitch) * self.R_v1v(yaw)
        return R

    def rotate(self, ypr_rad):
        vertices = self.vertices * self.R_bv(ypr_rad)
        return vertices
    
    def update_view(self):
        vertices = self.rotate(self.dynamics.x[6:9]) + self.dynamics.x[0:3]
        self.viewer.update(vertices)
    
    def update_state(self, dt):
         if self.dynamics.control_inputs is not None:
             t = dt + self.dynamics.integrator.t
             self.dynamics.integrate(t)
         else:
             raise Exception('set control inputs first')
         
    def set_state(self, x, t):
        self.dynamics.integrator.set_initial_value(x, t)
        
    def set_control_inputs(self, control_inputs):
        self.dynamics.control_inputs = control_inputs
        
#    def __call__(self, t):
#        self.dynamics.integrate(t)
#        self.x = self.dynamics.integrator.y    