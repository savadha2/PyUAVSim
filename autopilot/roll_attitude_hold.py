# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:40:46 2015

@author: sharath
"""
from controllers.pid import PID
import yaml

class RollAttitudeHold(PID):
    def __init__(self, config, Ts):
        self.kp = 1
        self.kd = 0
        self.ki = 0
        self.limit = 0
        self.tau = 0
        super(RollAttitudeHold, self).__init__(self.kp, self.ki, self.kd, self.limit, Ts, self.tau)
    
    def delta_a(self, phi_c, phi):
        return self.compute_control_input(phi_c, phi)

    
