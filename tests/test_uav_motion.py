import numpy as np
import mpl_toolkits.mplot3d as a3
import pylab as pl
from fixed_wing import FixedWingUAV#, FixedWingUAVDynamics

ax = a3.Axes3D(pl.figure(1))
ax.set_xlim3d(-20, 20)
ax.set_ylim3d(-20, 20)
ax.set_zlim3d(0, 40)
#initial_state = [0, 0, 0, 20., 0., 0.0, 0, 0 * np.pi/180, 0, 0, 0, 0.2]
#initial_state = [0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#         1.99981082e+01,  -1.90210767e-05,  -2.75077318e-01,
#        -1.06772734e-07,  -1.37542996e-02,   0.00000000e+00,
#         0.00000000e+00,  -0.00000000e+00,   0.00000000e+00]#np.array(([0, 0, 0]), dtype = np.double)
initial_state = [ 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         9.99976086e+00,  -3.08655613e-05,   6.91579462e-02,
        -4.68330235e-06,   6.91584975e-03,   0.00000000e+00,
        -0.00000000e+00,  -0.00000000e+00,   0.00000000e+00]
uav = FixedWingUAV(initial_state, 0, '../configs/zagi.yaml', ax)
pl.show()
t = uav.dynamics.t0
x, actuator_commands = uav.dynamics.compute_trimmed_states_inputs(20., 0., np.inf, 0., 0., 0.)
for m in range(2400):
    uav.update_state(dt = 1/200.)
    if m%25==0:
        uav.update_view()
        #uav2.update_view()
        print 'z speed: ', -uav.x[2]
        pl.pause(.01)