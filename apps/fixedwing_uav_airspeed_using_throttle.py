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
import mpl_toolkits.mplot3d as a3
import pylab as pl
from uav.fixed_wing import FixedWingUAV#, FixedWingUAVDynamics
from uav.autopilot import Autopilot
from viewer.viewer import UAVViewer

class AppFixedWingAirspeedHolder(FixedWingUAV):
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
        super(AppFixedWingAirspeedHolder, self).__init__(x0, t0, config)
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
        
    def update_view(self):
        vertices = self.rotate(self.dynamics.x[6:9]) + self.dynamics.x[0:3]
        self.viewer.update(vertices)
    
    def trim(self, Va, gamma, radius, max_iters):
        trimmed_state, trimmed_control_inputs = self.dynamics.trim(Va, gamma, radius, epsilon=1e-8, kappa=1e-6, max_iters=max_iters)
        #self.set_state(trimmed_state, 0.)
        #print trimmed_control_inputs
        self.set_control_inputs(trimmed_control_inputs)
    
    #TODO: vg = va (no wind is assumed. Add wind models and change appropriately later!)
    def set_airspeed(self, Va_c, zeta, Va_trim, delta_e_trim, alpha_trim, delta_t_trim):
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
        kp_v = (2.0 * zeta * omega_v - av_1) /av_2
        ki_v = omega_v**2/av_2
        self.autopilot.airspeed_hold_with_throttle_controller.kp = kp_v
        self.autopilot.airspeed_hold_with_pitch_controller.ki = ki_v
        delta_delta_t = self.autopilot.compute_throttle_for_airspeed(Va_c, Va)
        control_inputs = self.get_control_inputs()
        control_inputs[3] = delta_delta_t + control_inputs[3]
        self.set_control_inputs(control_inputs)
        
    
ax = a3.Axes3D(pl.figure(1))
ax.set_xlim3d(-20, 20)
ax.set_ylim3d(-20, 20)
ax.set_zlim3d(0, 40)
Va_trim = 35.0
initial_state = [0, 0, 0, Va_trim, 0., 0.0, 0, 0 * np.pi/180, 0, 0, 0, 0.2]
uav = AppFixedWingAirspeedHolder(initial_state, 0, '../configs/aerosonde.yaml', ax)
uav.trim(35., 0., np.inf, 5000)
delta_e_trim = uav.get_control_inputs()[0]
delta_t_trim = uav.get_control_inputs()[3]
alpha_trim = np.arctan(initial_state[5]/initial_state[3])
npoints = 2400
x = np.zeros((npoints, 12), dtype = np.double)
gamma = np.zeros((npoints,), dtype = np.double)
t = np.zeros((npoints,), dtype = np.double)

airspeed_command_history = np.zeros((npoints,), dtype = np.double)
airspeed_command =  38
pitch_command = 0
pl.show()
for m in range(npoints):
    if m%1000 == 0:
        airspeed_command = airspeed_command  #-altitude_command 
        print 'airspeed command: ', airspeed_command
    if m%1 == 0:
        uav.set_airspeed(airspeed_command, 0.5, Va_trim, delta_e_trim, alpha_trim, delta_t_trim)
    airspeed_command_history[m] = airspeed_command
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
airspeed = v
pl.plot(t, airspeed_command_history, '-r')
pl.plot(t, airspeed, '.b')
pl.xlabel('time (seconds)')
pl.ylabel('altitude (meters)')
pl.grid('on')
pl.legend(['set point', 'actual'])