# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:40:46 2015

@author: sharath
"""
from controllers.pid import PID as pid_controller
import yaml

class PID:
    def __init__(self, config, Ts):
        self.kp_phi = 1
        self.kd_phi = 0
        self.ki_phi = 0
        self.phi_limit = 0
        self.tau = 0
        self.roll_attitude_hold = pid_controller(self.kp_phi, self.ki_phi, self.kd, self.phi_limit, Ts, self.tau)
    
    def roll_attitude_hold(self, phi_c, phi):
        delta_a = self.roll_attitude_hold.compute_control_input(phi_c, phi)
        return delta_a
    