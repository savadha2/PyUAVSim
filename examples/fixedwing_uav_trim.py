import numpy as np
import mpl_toolkits.mplot3d as a3
import pylab as pl
from uav.fixed_wing import FixedWingUAV#, FixedWingUAVDynamics

ax = a3.Axes3D(pl.figure(1))
ax.set_xlim3d(-20, 20)
ax.set_ylim3d(-20, 20)
ax.set_zlim3d(0, 40)
initial_state = [0, 0, 0, 20., 0., 0.0, 0, 0 * np.pi/180, 0, 0, 0, 0.2]
uav = FixedWingUAV('../configs/zagi.yaml', ax)
trimmed_state, trimmed_control_inputs = uav.dynamics.trim(10., 0.0, -50., epsilon=1e-8, kappa=1e-6, max_iters=25000)
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
t = np.zeros((2400,), dtype = np.double)

for m in range(npoints):
    uav.update_state(dt = 1/200.)
    v[m] = np.linalg.norm(uav.dynamics.x[3:6])
    x[m] = uav.dynamics.x[0]
    y[m] = uav.dynamics.x[1]
    z[m] = uav.dynamics.x[2]
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