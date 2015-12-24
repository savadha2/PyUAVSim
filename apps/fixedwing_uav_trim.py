import numpy as np
import mpl_toolkits.mplot3d as a3
import pylab as pl
from uav.fixed_wing import FixedWingUAV#, FixedWingUAVDynamics
from viewer.viewer import UAVViewer

class AppFixedWingUAVTrim(FixedWingUAV):
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
        super(AppFixedWingUAVTrim, self).__init__(x0, t0, config)
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
        
    def update_view(self):
        vertices = self.rotate(self.dynamics.x[6:9]) + self.dynamics.x[0:3]
        self.viewer.update(vertices)
    
    def trim(self, Va, gamma, radius, max_iters):
        trimmed_state, trimmed_control_inputs = self.dynamics.trim(Va, gamma, radius, epsilon=1e-8, kappa=1e-6, max_iters=max_iters)
        self.set_state(trimmed_state, 0.)
        self.set_control_inputs(trimmed_control_inputs)
    
ax = a3.Axes3D(pl.figure(1))
ax.set_xlim3d(-20, 20)
ax.set_ylim3d(-20, 20)
ax.set_zlim3d(0, 40)
initial_state = [0, 0, 0, 35., 0., 0.0, 0, 0 * np.pi/180, 0, 0, 0, 0.2]
uav = AppFixedWingUAVTrim(initial_state, 0, '../configs/aerosonde.yaml', ax)
uav.trim(35., 0., 100, 25000)

npoints = 2400
x = np.zeros((2400, 12), dtype = np.double)
gamma = np.zeros((2400,), dtype = np.double)
t = np.zeros((2400,), dtype = np.double)

pl.show()
for m in range(npoints):
    uav.update_state(dt = 1/200.)
    v = np.linalg.norm(uav.dynamics.x[3:6])
    x[m, 0:12] = uav.dynamics.x[0:12]
    gamma[m] = np.arcsin(-uav.dynamics.x[5]/v)
    t[m] = uav.dynamics.t
    if m%25==0:
        uav.update_view()
        pl.pause(.01)
        
        
pl.figure(2)
pl.subplot(221)
east = x[:, 1]
north = x[:, 0]
pl.plot(east, north, 'r')
pl.axis('equal')
pl.xlabel('East (m)')
pl.ylabel('North (m)')
pl.grid('on')

pl.subplot(222)
z = x[:, 2]
pl.plot(t, -z, '.g')
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
pl.subplot(131)
pl.plot(t, x[:, 6] * 180./np.pi)
pl.ylabel('roll (deg)')
pl.xlabel('time (secs)')

pl.subplot(132)
pl.plot(t, x[:, 7] * 180./np.pi)
pl.ylabel('pitch (deg)')
pl.xlabel('time (secs)')

pl.subplot(133)
pl.plot(t, x[:, 8] * 180./np.pi)
pl.ylabel('yaw (deg)')
pl.xlabel('time (secs)')