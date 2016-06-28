# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 07:28:12 2016

@author: sharath
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 07:33:45 2016

@author: sharath
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:32:53 2015

@author: sharath
"""
import numpy as np
from uav.fixed_wing import FixedWingUAV#, FixedWingUAVDynamics
from uav.autopilot import Autopilot
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


class AppFixedWingUAVAutopilot(FixedWingUAV, Autopilot):
    def __init__(self, x0, t0, config, ax):
        FixedWingUAV.__init__(self, x0, t0, config)
        Autopilot.__init__(self, self.attrs['autopilot'], 1./200.)        
        self.x0 = x0
        self.viewer = UAVViewer(ax, x0, self.R_bv(x0[6:9]))        
        #self = Autopilot(self.attrs['autopilot'], 1./200.)        
        
    def update_view(self):
        new_vertices = self.viewer.rotate(self.R_bv(self.dynamics.x[6:9])) + self.dynamics.x[0:3] - self.x0[0:3]
        self.viewer.update(new_vertices)
    
    def trim(self, Va, gamma, radius, max_iters):
        trimmed_state, trimmed_control_inputs = self.dynamics.trim(Va, gamma, radius, epsilon=1e-8, kappa=1e-6, max_iters=max_iters)
        self.set_control_inputs(trimmed_control_inputs)
        
    def get_roll_for_heading(self, chi_c):
        Va = np.linalg.norm(self.dynamics.x[3:6])
        omega_chi = 0.1
        Vg = Va
        self.heading_hold_controller.kp = 2 * 3.0 * omega_chi * Vg/9.81
        self.heading_hold_controller.ki = omega_chi**2 * Vg/9.81
        chi = self.dynamics.x[8]
        roll_c = self.compute_roll(chi_c, chi)
        return roll_c
        
    def set_roll(self, roll_c):        
        Va = np.linalg.norm(self.dynamics.x[3:6])        
        S = self.attrs['params']['S']
        b = self.attrs['params']['b']
        rho = self.attrs['params']['rho']
        Jx = self.attrs['params']['Jx']
        Jz = self.attrs['params']['Jz']
        Jxz = self.attrs['params']['Jxz']
        gamma_0 = Jx * Jz - Jxz**2
        gamma_3 = Jz/gamma_0
        gamma_4 = Jxz/gamma_0
        Clateral_coeffs = self.attrs['lateral_coeffs']
        Cl_p = Clateral_coeffs['Cl_p']
        Cl_delta_a = Clateral_coeffs['Cl_delta_a']
        Cn_delta_a = Clateral_coeffs['Cn_delta_a']
        Cn_p = Clateral_coeffs['Cn_p']
        Cp_delta_a = gamma_3 * Cl_delta_a + gamma_4 * Cn_delta_a
        Cp_p = gamma_3 * Cl_p + gamma_4 * Cn_p
        aphi_1 = -0.5 * rho * Va * S * b * Cp_p * 0.5 * b
        aphi_2 = 0.5 * rho *  Va**2 * S * b * Cp_delta_a
        self.roll_hold_controller.kp = self.config['delta_a_max_deg']/self.config['error_phi_max_deg'] * np.sign(aphi_2)
        self.roll_hold_controller.ki = self.config['roll']['ki']
        omega_phi = np.sqrt(np.abs(aphi_2) * self.config['delta_a_max_deg']/self.config['error_phi_max_deg'])        
        #print 'omega_phi: ', omega_phi
        zeta = self.config['roll']['zeta']
        self.roll_hold_controller.kd = (2 * zeta * omega_phi - aphi_1)/aphi_2
        self.roll_hold_controller.tau = self.config['roll']['tau']
        control_inputs = self.get_control_inputs()
        #-0.018949908534872429
        control_inputs[1] = self.compute_delta_a(roll_c, self.dynamics.x[6])
        self.set_control_inputs(control_inputs)        
    
    def get_throttle_for_airspeed(self, Va_c, Va_trim, delta_e_trim, alpha_trim, delta_t_trim):
        Va = np.linalg.norm(self.dynamics.x[3:6])
        S = self.attrs['params']['S']
        rho = self.attrs['params']['rho']
        Clong_coeffs = self.attrs['longitudinal_coeffs']
        CD0 = Clong_coeffs['CD0']
        CD_alpha = Clong_coeffs['CD_alpha']
        CD_delta_e = Clong_coeffs['CD_delta_e']
        C_prop = Clong_coeffs['C_prop']
        S_prop = self.attrs['params']['S_prop']
        mass = self.attrs['params']['mass']
        k_motor = self.attrs['params']['k_motor']        
        av_1 = rho * Va_trim * S * (CD0 + CD_alpha * alpha_trim + CD_delta_e * delta_e_trim)/mass + rho * S_prop * C_prop * Va_trim/mass
        av_2 = rho * S_prop * C_prop * k_motor**2 * delta_t_trim/mass
        omega_v = 1.0
        zeta = self.config['airspeed']['throttle']['zeta']
        kp_v = (2.0 * zeta * omega_v - av_1) /av_2
        ki_v = omega_v**2/av_2
        self.airspeed_hold_with_throttle_controller.kp = kp_v
        self.airspeed_hold_with_pitch_controller.ki = ki_v
        delta_delta_t = self.compute_throttle_for_airspeed(Va_c, Va)
        return delta_delta_t + self.get_control_inputs()[3]
        
    def get_pitch_for_altitude(self, h_c):
        Va = np.linalg.norm(self.dynamics.x[3:6])
        S = self.attrs['params']['S']
        #b = self.attrs['params']['b']
        c = self.attrs['params']['c']
        rho = self.attrs['params']['rho']
        Jy = self.attrs['params']['Jy']        
        Clong_coeffs = self.attrs['longitudinal_coeffs']
        Cm_alpha = Clong_coeffs['Cm_q']
        Cm_delta_e = Clong_coeffs['Cm_delta_e']
        atheta_2 =  -rho * Va**2 * c *S * Cm_alpha * 0.5/Jy
        atheta_3 = rho * Va**2 * c *S * Cm_delta_e * 0.5/Jy
        omega_h = 0.1
        kp_theta = self.config['delta_e_max_deg']/self.config['error_theta_max_deg'] * np.sign(atheta_3)
        K_theta_dc = (kp_theta * atheta_3)/(atheta_2 + kp_theta * atheta_3)
        zeta = self.config['altitude']['pitch']['zeta']
        self.altitude_hold_controller.kp = 2.0 * zeta * omega_h/(K_theta_dc * Va)
        self.heading_hold_controller.ki = omega_h**2 /(K_theta_dc * Va)
        h = -self.dynamics.x[2]
        pitch_c = self.compute_pitch(h_c, h)
        return pitch_c
        
    def get_pitch_for_airspeed(self, Va_c, Va_trim, delta_e_trim, alpha_trim):
        Va = np.linalg.norm(self.dynamics.x[3:6])
        S = self.attrs['params']['S']
        #b = self.attrs['params']['b']
        c = self.attrs['params']['c']
        rho = self.attrs['params']['rho']
        Jy = self.attrs['params']['Jy']        
        Clong_coeffs = self.attrs['longitudinal_coeffs']
        CD0 = Clong_coeffs['CD0']
        CD_alpha = Clong_coeffs['CD_alpha']
        CD_delta_e = Clong_coeffs['CD_delta_e']
        C_prop = Clong_coeffs['C_prop']
        S_prop = self.attrs['params']['S_prop']
        mass = self.attrs['params']['mass']
        aV_1 = rho * Va_trim * S * (CD0 + CD_alpha * alpha_trim + CD_delta_e * delta_e_trim)/mass + rho * S_prop * C_prop * Va_trim/mass
        Cm_alpha = Clong_coeffs['Cm_q']
        Cm_delta_e = Clong_coeffs['Cm_delta_e']
        atheta_2 =  -rho * Va**2 * c *S * Cm_alpha * 0.5/Jy
        atheta_3 = rho * Va**2 * c *S * Cm_delta_e * 0.5/Jy
        omega_v2 = 0.5
        kp_theta = self.config['delta_e_max_deg']/self.config['error_theta_max_deg'] * np.sign(atheta_3)
        K_theta_dc = (kp_theta * atheta_3)/(atheta_2 + kp_theta * atheta_3)
        zeta = self.config['airspeed']['pitch']['zeta']
        self.airspeed_hold_with_pitch_controller.kp = (aV_1 - 2.0*zeta*omega_v2)/(K_theta_dc * 9.81)
        self.airspeed_hold_with_pitch_controller.ki = -omega_v2**2 /(K_theta_dc * 9.81)
        pitch_c = self.compute_pitch_for_airspeed(Va_c, Va)
        return pitch_c
        
    def set_pitch(self, pitch_c):
        Va = np.linalg.norm(self.dynamics.x[3:6])
        S = self.attrs['params']['S']
        #b = self.attrs['params']['b']
        c = self.attrs['params']['c']
        rho = self.attrs['params']['rho']
        Jy = self.attrs['params']['Jy']        
        Clong_coeffs = self.attrs['longitudinal_coeffs']
        Cm_q = Clong_coeffs['Cm_q']
        Cm_alpha = Clong_coeffs['Cm_q']
        Cm_delta_e = Clong_coeffs['Cm_delta_e']
        atheta_1 = -rho * Va * c * S * Cm_q * 0.5 * c
        atheta_2 =  -rho * Va**2 * c *S * Cm_alpha * 0.5/Jy
        atheta_3 = rho * Va**2 * c *S * Cm_delta_e * 0.5/Jy
        self.pitch_hold_controller.kp = self.config['delta_e_max_deg']/self.config['error_theta_max_deg'] * np.sign(atheta_3)
        self.pitch_hold_controller.ki = self.config['pitch']['ki']
        omega_theta = np.sqrt(atheta_2 + self.config['delta_e_max_deg']/self.config['error_theta_max_deg'] * np.abs(atheta_3))
        zeta = self.config['pitch']['zeta']
        self.pitch_hold_controller.kd = (2 * zeta * omega_theta - atheta_1)/atheta_3
        self.pitch_hold_controller.tau = self.config['pitch']['tau']
        control_inputs = self.get_control_inputs()
        q = self.dynamics.x[10]
        control_inputs[0] = self.compute_delta_e(pitch_c, self.dynamics.x[7], -q)
        self.set_control_inputs(control_inputs)
        
    def set_throttle(self, throttle_c):
        control_inputs = self.get_control_inputs()
        control_inputs[3] = throttle_c
        self.set_control_inputs(control_inputs)
        
    def __call__(self, Va_c, chi_c, h_c, trimmed_state, trimmed_control, h_takeoff, h_hold):
        roll_c = self.get_roll_for_heading(chi_c)
        self.set_roll(roll_c)
        delta_e_trim = trimmed_control[0]
        delta_t_trim = trimmed_control[3]
        Va_trim = np.linalg.norm(trimmed_state[0:3])
        alpha_trim = np.arctan(trimmed_state[5]/trimmed_state[3])
        #self.set_airspeed_with_throttle(Va_c, Va_trim, delta_e_trim, alpha_trim, delta_t_trim)
        #self.set_airspeed_with_pitch(Va_c, Va_trim, delta_e_trim, alpha_trim)
        h = -self.x[2]
        if h<h_takeoff:
            self.set_pitch(self.config['delta_e_max_deg'] * np.pi/180.)
            self.set_throttle(1.0)
        elif h>=h_takeoff and h<h_c- h_hold:
            pitch_c = self.get_pitch_for_airspeed(Va_c, Va_trim, delta_e_trim, alpha_trim)
            self.set_pitch(pitch_c)
            self.set_throttle(1.0)
        elif h<h_c+h_hold and h>=h_c-h_hold:
            throttle_c = self.get_throttle_for_airspeed(Va_c, Va_trim, delta_e_trim, alpha_trim, delta_t_trim)
            self.set_throttle(throttle_c)
            pitch_c = self.get_pitch_for_altitude(h_c)
            self.set_pitch(pitch_c)
        elif h>=h_c+h_hold:
            pitch_c = self.get_pitch_for_airspeed(Va_c, Va_trim, delta_e_trim, alpha_trim)
            self.set_pitch(pitch_c)
            self.set_throttle(0.0)            
        
        