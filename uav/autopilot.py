# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:40:46 2015

@author: sharath
"""
from controllers.pid import PID

class Autopilot:
    def __init__(self, attrs, Ts):
        self.kp_phi = 1
        self.kd_phi = 0
        self.ki_phi = 0
        self.delta_a_limit = 0
        self.kd_phi_tau = 0
        self.roll_hold_controller = PID(self.kp_phi, self.ki_phi, self.kd_phi, self.delta_a_limit, Ts, self.kd_phi_tau)
    
    def compute_delta_a(self, phi_c, phi):
        return self.roll_hold_controller.compute_control_input(phi_c, phi)

    
