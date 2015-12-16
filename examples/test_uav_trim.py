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
for m in range(2400):
    uav.update_state(dt = 1/200.)
    if m%25==0:
        uav.update_view()
        pl.pause(.01)
