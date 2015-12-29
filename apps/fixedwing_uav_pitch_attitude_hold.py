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
        self.autopilot.delta_e_limit = self.attrs['autopilot']['delta_e_max_deg'] * np.pi/180
        self.autopilot.pitch_hold_controller.limit = self.autopilot.delta_e_limit
        
    def update_view(self):
        vertices = self.rotate(self.dynamics.x[6:9]) + self.dynamics.x[0:3]
        self.viewer.update(vertices)
    
    def trim(self, Va, gamma, radius, max_iters):
        trimmed_state, trimmed_control_inputs = self.dynamics.trim(Va, gamma, radius, epsilon=1e-8, kappa=1e-6, max_iters=max_iters)
        #self.set_state(trimmed_state, 0.)
        self.set_control_inputs(trimmed_control_inputs)
    
    #TODO: move ki, tau, zeta to config if required
    def set_pitch(self, pitch_c, ki, tau, zeta):
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
        self.autopilot.pitch_hold_controller.kp = self.attrs['autopilot']['delta_e_max_deg']/self.attrs['autopilot']['error_theta_max_deg'] * np.sign(atheta_3)
        self.autopilot.pitch_hold_controller.ki = ki
        omega_theta = np.sqrt(atheta_2 + self.attrs['autopilot']['delta_e_max_deg']/self.attrs['autopilot']['error_theta_max_deg'] * np.abs(atheta_3))
        self.autopilot.pitch_hold_controller.kd = (2 * zeta * omega_theta - atheta_1)/atheta_3
        self.autopilot.pitch_hold_controller.tau = tau
        control_inputs = self.get_control_inputs()
        #-0.018949908534872429
        q = self.dynamics.x[10]
        control_inputs[0] = self.autopilot.compute_delta_e(pitch_c, self.dynamics.x[7], -q)
        self.set_control_inputs(control_inputs)
    
ax = a3.Axes3D(pl.figure(1))
ax.set_xlim3d(-20, 20)
ax.set_ylim3d(-20, 20)
ax.set_zlim3d(0, 40)
initial_state = [0, 0, 0, 35., 0., 0.0, 0, 0 * np.pi/180, 0, 0, 0, 0.2]
uav = AppFixedWingRollAttHolder(initial_state, 0, '../configs/aerosonde.yaml', ax)
uav.trim(35., 0., np.inf, 5000)

npoints = 2400
x = np.zeros((npoints, 12), dtype = np.double)
gamma = np.zeros((npoints,), dtype = np.double)
t = np.zeros((npoints,), dtype = np.double)

pitch_command_history = np.zeros((npoints,), dtype = np.double)
pitch_command = 5 * np.pi/180

pl.show()
for m in range(npoints):
    if m%400 == 0:
        pitch_command = -pitch_command 
        print 'pitch command: ', pitch_command
    pitch_command_history[m] = pitch_command
    uav.set_pitch(pitch_command, 0, .05, 0.7)    
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
pitch = x[:, 7]
pl.plot(t, pitch_command_history * 180/np.pi, '-r')
pl.plot(t, pitch * 180/np.pi, '.b')
pl.xlabel('time (seconds)')
pl.ylabel('pitch (degrees)')
pl.grid('on')
pl.legend(['set point', 'actual'])