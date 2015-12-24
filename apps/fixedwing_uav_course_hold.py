# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:32:53 2015

@author: sharath
"""
import numpy as np
import mpl_toolkits.mplot3d as a3
import pylab as pl
from uav.fixed_wing import FixedWingUAV#, FixedWingUAVDynamics
from uav.autopilot import Autopilot
from viewer.viewer import UAVViewer

class AppFixedWingRollAttHolder(FixedWingUAV):
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
    def __init__(self, x0, t0, config, ax):
        super(AppFixedWingRollAttHolder, self).__init__(x0, t0, config)
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
        
        self.nose = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 1, 4]]
        self.fuselage = [[5, 3, 2], [5, 2, 1], [5, 3, 4], [5, 4, 1]]
        self.wing = [[6, 7, 8, 9]]
        self.tail_wing = [[10, 11, 12, 13]]
        self.tail = [[5, 14, 15]]
        self.faces = [self.nose, self.fuselage, self.wing, self.tail_wing, self.tail]
        self.viewer = UAVViewer(ax, (self.rotate(x0[6:9]) + x0[0:3]), self.faces, ['r', 'g', 'g', 'g', 'y'])
        self.autopilot = Autopilot([], 1./200.)
        self.autopilot.delta_a_limit = 15. * np.pi/180.
        self.autopilot.roll_hold_controller.limit = self.autopilot.delta_a_limit
        
    def update_view(self):
        vertices = self.rotate(self.dynamics.x[6:9]) + self.dynamics.x[0:3]
        self.viewer.update(vertices)
    
    def trim(self, Va, gamma, radius, max_iters):
        trimmed_state, trimmed_control_inputs = self.dynamics.trim(Va, gamma, radius, epsilon=1e-8, kappa=1e-6, max_iters=max_iters)
        #self.set_state(trimmed_state, 0.)
        #print trimmed_control_inputs
        self.set_control_inputs(trimmed_control_inputs)
    
    #TODO: vg = va (no wind is assumed. Add wind models and change appropriately later!)
    def set_heading(self, chi_c):
        Va = np.linalg.norm(self.dynamics.x[3:6])
        omega_chi = 0.5
        Vg = Va
        self.autopilot.heading_hold_controller.kp = 2 * 1.5 * omega_chi * Vg/9.81
        self.autopilot.heading_hold_controller.ki = omega_chi**2 * Vg/9.81
        chi = self.dynamics.x[8]
        roll_c = self.autopilot.compute_roll(chi_c, chi)
        return roll_c
        
    def set_roll(self, roll_c, ki, tau, zeta):
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
        self.autopilot.roll_hold_controller.kp = self.attrs['autopilot']['delta_a_max_deg']/self.attrs['autopilot']['error_phi_max_deg'] * np.sign(aphi_2)
        self.autopilot.roll_hold_controller.ki = ki
        omega_phi = np.sqrt(np.abs(aphi_2) * self.attrs['autopilot']['delta_a_max_deg']/self.attrs['autopilot']['error_phi_max_deg'])        
        #print 'omega_phi: ', omega_phi
        self.autopilot.roll_hold_controller.kd = (2 * zeta * omega_phi - aphi_1)/aphi_2
        self.autopilot.roll_hold_controller.tau = tau
        control_inputs = self.get_control_inputs()
        #-0.018949908534872429
        control_inputs[1] = self.autopilot.compute_delta_a(roll_c, self.dynamics.x[6])
        self.set_control_inputs(control_inputs)
    
ax = a3.Axes3D(pl.figure(1))
ax.set_xlim3d(-20, 20)
ax.set_ylim3d(-20, 20)
ax.set_zlim3d(0, 40)
initial_state = [0, 0, 0, 35., 0., 0.0, 0, 0 * np.pi/180, 5 * np.pi/180., 0, 0, 0.0]
uav = AppFixedWingRollAttHolder(initial_state, 0, '../configs/aerosonde.yaml', ax)
uav.trim(35., 0., np.inf, 5000)

npoints = 2400
x = np.zeros((2400, 12), dtype = np.double)
gamma = np.zeros((2400,), dtype = np.double)
chi = np.zeros((2400,), dtype = np.double)
t = np.zeros((2400,), dtype = np.double)

chi_command_history = np.zeros((2400,), dtype = np.double)
chi_command = 10 * np.pi/180
roll_c = 0
pl.show()
for m in range(npoints):
    if m%400 == 0:
        chi_command = chi_command 
        print 'chi command: ', chi_command
    if m%50 == 0:
        roll_c = uav.set_heading(chi_command)
    velocity_inertial = np.asarray(uav.dynamics.x[3:6]) * uav.R_bv(uav.dynamics.x[6:9])
    chi[m] = uav.dynamics.x[8]#np.arctan2(velocity_inertial[0, 1], velocity_inertial[0, 0])
    chi_command_history[m] = chi_command
    uav.set_roll(roll_c, 5, .05, 1.5)    
    uav.update_state(dt = 1/200.)
    v = np.linalg.norm(uav.dynamics.x[3:6])
    x[m, :] = uav.dynamics.x
    gamma[m] = np.arcsin(-uav.dynamics.x[5]/v)
    t[m] = uav.dynamics.t
    if m%25==0:
        uav.update_view()
        pl.pause(.01)   
        
pl.figure(2)
pl.subplot(221)
north = x[:, 0]
east = x[:, 1]
pl.plot(east, north, 'r')
pl.axis('equal')
pl.xlabel('East (m)')
pl.ylabel('North (m)')
pl.grid('on')

pl.subplot(222)
down = x[:, 2]
z = -down
pl.plot(t, z, '.g')
pl.grid
pl.ylabel('Altitude (m)')
pl.xlabel('time (seconds)')
pl.grid('on')

pl.subplot(223)
v = np.linalg.norm(x[:, 3:6], axis=1)
pl.plot(t, v)
pl.xlabel('time (seconds)')
pl.ylabel('air speed (m/s)')
pl.grid('on')

pl.subplot(224)
pl.plot(t, gamma * 180./np.pi)
pl.xlabel('time (seconds)')
pl.ylabel('gamma (deg)')
pl.grid('on')

pl.figure(3)
pl.plot(t, chi_command_history * 180/np.pi, '-r')
pl.plot(t, chi * 180/np.pi, '.b')
pl.xlabel('time (seconds)')
pl.ylabel('roll (degrees)')
pl.grid('on')
pl.legend(['set point', 'actual'])