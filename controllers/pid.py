# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 14:56:35 2015

@author: sharath
"""
import numpy as np
class PID(object):
    def __init__(self, kp, ki, kd, limit, Ts, tau):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self.Ts = Ts
        self.tau = tau
        self.integrator = 0.
        self.differentiator= 0.
        self.error_d1 = 0.
    
    def compute_control_input(self, y_c, y):
        error = y_c - y        
        self.integrator += 0.5 * self.Ts * (error + self.error_d1)
        #band limited differentiator
        self.differentiator = (2*self.tau - self.Ts)/(2*self.tau - self.Ts) * self.differentiator \
                                + 2/(2 * self.tau + self.Ts) * (error - self.error_d1)
        self.error_d1 = error
        u_unsat = self.kp * error + self.ki * self.integrator + self.kd * self.differentiator
        u = np.sign(u_unsat) * np.max([np.abs(u_unsat), self.limit])
        #integrator anti wind up
        if self.ki!=0:
            self.integrator += self.Ts/self.ki * (u - u_unsat)
        return u
            