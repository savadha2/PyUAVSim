# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:32:53 2015

@author: sharath
"""
import numpy as np
import mpl_toolkits.mplot3d as a3
import pylab as pl
from uav.fixed_wing import FixedWingUAV#, FixedWingUAVDynamics
from autopilot.roll_attitude_hold import RollAttitudeHold

ax = a3.Axes3D(pl.figure(1))
ax.set_xlim3d(-20, 20)
ax.set_ylim3d(-20, 20)
ax.set_zlim3d(0, 40)
uav = FixedWingUAV('../configs/zagi.yaml', ax)
trimmed_state, trimmed_control_inputs = uav.dynamics.trim(10., 0.0, np.inf, epsilon=1e-8, kappa=1e-6, max_iters=5000)
uav.set_state(trimmed_state, 0.)
uav.set_control_inputs(trimmed_control_inputs)
t = uav.dynamics.t0
pl.show()
npoints = 2400
v = np.zeros((2400,), dtype = np.double)
gamma = np.zeros((2400,), dtype = np.double)
x = np.zeros((2400,), dtype = np.double)
y = np.zeros((2400,), dtype = np.double)
z = np.zeros((2400,), dtype = np.double)
roll = np.zeros((2400,), dtype = np.double)
t = np.zeros((2400,), dtype = np.double)

roll_command_history = np.zeros((2400,), dtype = np.double)
roll_command = 10 * np.pi/180
roll_attitude_holder = RollAttitudeHold([], 1./200.)
roll_attitude_holder.kp = 6
roll_attitude_holder.ki = 5
roll_attitude_holder.kd = .6
roll_attitude_holder.tau = .05
roll_attitude_holder.limit = 15. * np.pi/180.
control_inputs = trimmed_control_inputs
for m in range(npoints):
    if m%400 == 0:
        roll_command = roll_command * -1
    roll_command_history[m] = roll_command
    control_inputs[1] = roll_attitude_holder.compute_control_input(roll_command, uav.dynamics.x[6])
    uav.set_control_inputs(control_inputs)
    uav.update_state(dt = 1/200.)
    v[m] = np.linalg.norm(uav.dynamics.x[3:6])
    x[m] = uav.dynamics.x[0]
    y[m] = uav.dynamics.x[1]
    z[m] = uav.dynamics.x[2]
    roll[m] = uav.dynamics.x[6]
    gamma[m] = np.arcsin(-uav.dynamics.x[5]/v[m])
    t[m] = uav.dynamics.t
    if m%25==0:
        uav.update_view()
        pl.pause(.01)
        
        
pl.figure(2)
pl.subplot(221)
pl.plot(y, x, 'r')
pl.axis('equal')
pl.xlabel('East (m)')
pl.ylabel('North (m)')
pl.grid('on')

pl.subplot(222)
pl.plot(t, -z, '.g')
pl.grid
pl.ylabel('Altitude (m)')
pl.xlabel('time (seconds)')
pl.grid('on')

pl.subplot(223)
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
pl.plot(t, roll_command_history * 180/np.pi, '-r')
pl.plot(t, roll * 180/np.pi, '.b')