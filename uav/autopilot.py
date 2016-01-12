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
        self.delta_e_limit = np.inf
        self.kd_phi_tau = 0
        self.roll_hold_controller = PID(self.kp_phi, self.ki_phi, self.kd_phi, self.delta_a_limit, Ts, self.kd_phi_tau)
        
        self.kp_chi = 0
        self.ki_chi = 0
        self.heading_hold_controller = PID(self.kp_chi, self.ki_chi, 0, np.inf, Ts * 1.0, 0)
        
        self.kp_theta = 0
        self.ki_theta = 0
        self.pitch_hold_controller = PID(self.kp_theta, self.ki_theta, 0, self.delta_e_limit, Ts, 0)
        
        self.kp_h = 1
        self.ki_h = 0
        self.altitude_hold_controller = PID(self.kp_h, self.ki_h, 0, np.inf, Ts * 1.0, 0)
        
        self.kp_v1 = 1
        self.ki_v1 = 0
        self.airspeed_hold_with_pitch_controller = PID(self.kp_v1, self.ki_v1, 0, np.inf, Ts, 0)
        
        self.kp_v = 1
        self.ki_v = 0
        self.airspeed_hold_with_throttle_controller = PID(self.kp_v, self.ki_v, 0, np.inf, Ts, 0)
    
    def compute_delta_a(self, phi_c, phi, *args):
        return self.roll_hold_controller.compute_control_input(phi_c, phi, *args)
    
    def compute_roll(self, chi_c, chi):
        return self.heading_hold_controller.compute_control_input(chi_c, chi)
        
    def compute_delta_e(self, theta_c, theta, *args):
        return self.pitch_hold_controller.compute_control_input(theta_c, theta, *args)
        
    def compute_pitch(self, h_c, h, *args):
        return self.altitude_hold_controller.compute_control_input(h_c, h, *args)
        
    def compute_pitch_for_airspeed(self, Va_c, Va, *args):
        return self.airspeed_hold_with_pitch_controller.compute_control_input(Va_c, Va, *args)
        
    def compute_throttle_for_airspeed(self, Va_c, Va, *args):
        return self.airspeed_hold_with_throttle_controller.compute_control_input(Va_c, Va, *args)


    
