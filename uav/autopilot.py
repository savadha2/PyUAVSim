# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:40:46 2015

@author: sharath
"""
from controllers.pid import PID
import numpy as np

class Autopilot:
    def __init__(self, attrs, Ts):
        self.kp_phi = 1
        self.kd_phi = 0
        self.ki_phi = 0
        self.delta_a_limit = np.inf
        self.kd_phi_tau = 0
        self.roll_hold_controller = PID(self.kp_phi, self.ki_phi, self.kd_phi, self.delta_a_limit, Ts, self.kd_phi_tau)
        
        self.kp_chi = 0
        self.ki_chi = 0
        self.heading_hold_controller = PID(self.kp_chi, self.ki_chi, 0, np.inf, Ts * 10., 0)
    
    def compute_delta_a(self, phi_c, phi, *args):
        return self.roll_hold_controller.compute_control_input(phi_c, phi, *args)
    
    def compute_roll(self, chi_c, chi):
        return self.heading_hold_controller.compute_control_input(chi_c, chi)

    
