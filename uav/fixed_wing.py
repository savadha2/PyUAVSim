import numpy as np
from dynamics import FixedWingUAVDynamics
import yaml
import errno
#import scipy as sp
#import matplotlib.colors as colors    
class FixedWingUAV(object):    
    def __init__(self, x0, t0, dynamics_config_file):
        self.attrs = None
        try:
            with open(dynamics_config_file) as f:
                self.attrs = yaml.load(f)
        except:
            raise IOError(errno.ENOENT, 'File not found', dynamics_config_file)
        self.dynamics = FixedWingUAVDynamics(x0, t0, 0.001, self.attrs)
    
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
        
    def get_control_inputs(self):
        return self.dynamics.control_inputs
        
#    def __call__(self, t):
#        self.dynamics.integrate(t)
#        self.x = self.dynamics.integrator.y    