# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 08:43:24 2016

@author: sharath

This is not unit testing!

"""
from apps.fixedwing_uav_autopilot import AppFixedWingUAVAutopilot
from apps.fixedwing_uav_trim import AppFixedWingUAVTrim
import numpy as np
import mpl_toolkits.mplot3d as a3
import pylab as pl

#@TODO: barebones right now, will add features and change interfaces as this evolves
class Tester:
    def __init__(self, x0, t0, config_path, ax, T_sim, dt = 1/200.0):
        self.x0 = x0
        self.t0 = t0
        self.config_path = config_path
        self.ax = ax
        self.T_sim = T_sim
        self.dt = dt
        self.npoints = int(T_sim/dt)        
        self.x = np.zeros((self.npoints, 12), dtype = np.double)
        self.alpha_beta_gamma = np.zeros((self.npoints, 3), dtype = np.double)
        self.t = np.zeros((self.npoints,), dtype = np.double)
        
    def test_trim(self, Va = 35, gamma = 0.025, R = 100, iters = 1500, animate = True):
        uav = AppFixedWingUAVTrim(self.x0, self.t0, self.config_path, self.ax)
        uav.trim(Va, 0.025, R, iters)
        # trim routine above sets the control inputs too unlike the other routines. Beware!
        for m in range(self.npoints):
            uav.update_state(dt = self.dt)
            self.x[m, 0:12] = uav.dynamics.x[0:12]
            self.alpha_beta_gamma[m, 0] = np.arctan(self.x[m , 5]/self.x[m , 3])
            v = np.linalg.norm(self.x[m, 3:6])
            self.alpha_beta_gamma[m, 1] = np.arcsin(self.x[m , 4]/v)
            theta = uav.dynamics.x[7]
            self.alpha_beta_gamma[m, 2] = theta - self.alpha_beta_gamma[m, 0]
            self.t[m] = uav.dynamics.t
            if m%5==0 and animate is True:
                uav.update_view()
                #pl.hold('on')
                self.ax.plot(self.x[0:m:5, 1], self.x[0:m:5, 0], -self.x[0:m:5, 2], '.g')
                pl.pause(.01)

    def test_autopilot(self, Va_trim = 70, gamma_trim = 0, R = np.inf, iters = 1500, animate = True):
        uav = AppFixedWingUAVAutopilot(self.x0, self.t0, self.config_path, self.ax)
        uav.trim(Va_trim, gamma_trim, R, iters)
        for m in range(self.npoints):    
            uav(Va_trim, 0, 500, 100, 50) #(Va, chi_c, h_c, h_takeoff, h_hold)
            uav.update_state(dt = self.dt)
            self.x[m, 0:12] = uav.dynamics.x[0:12]            
            self.alpha_beta_gamma[m, 0] = np.arctan(self.x[m , 5]/self.x[m , 3])
            v = np.linalg.norm(self.x[m, 3:6])
            self.alpha_beta_gamma[m, 1] = np.arcsin(self.x[m , 4]/v)
            theta = uav.dynamics.x[7]
            self.alpha_beta_gamma[m, 2] = theta - self.alpha_beta_gamma[m, 0]
            self.t[m] = uav.dynamics.t
            if m%5==0 and animate is True:
                uav.update_view()
                #pl.hold('on')
                self.ax.plot(self.x[0:m:5, 1], self.x[0:m:5, 0], -self.x[0:m:5, 2], '.g')
                pl.pause(.01)
    
    def plot(self):
        x = self.x
        alpha = self.alpha_beta_gamma[:, 0]
        beta = self.alpha_beta_gamma[:, 1]
        gamma = self.alpha_beta_gamma[:, 2]
        t = self.t
        
        pl.show()                          
        pl.figure(2)
        pl.subplot(331)
        east = x[:, 1]
        north = x[:, 0]
        pl.plot(east, north, 'r')
        pl.axis('equal')
        pl.xlabel('East (m)')
        pl.ylabel('North (m)')
        pl.grid('on')
        
        pl.subplot(332)
        z = x[:, 2]
        pl.plot(t, -z, '.g')
        pl.grid
        pl.ylabel('Altitude (m)')
        pl.xlabel('time (seconds)')
        pl.grid('on')
        
        pl.subplot(333)
        v = np.linalg.norm(x[:, 3:6], axis=1)
        pl.plot(t, v)
        pl.xlabel('time (seconds)')
        pl.ylabel('air speed (m/s)')
        pl.grid('on')
        
        pl.subplot(334)
        pl.plot(t, x[:, 6] * 180./np.pi)
        pl.ylabel('roll (deg)')
        pl.xlabel('time (secs)')
        pl.grid('on')
        
        pl.subplot(335)
        pl.plot(t, x[:, 7] * 180./np.pi)
        pl.ylabel('pitch (deg)')
        pl.xlabel('time (secs)')
        pl.grid('on')        

        pl.subplot(336)
        pl.plot(t, x[:, 8] * 180./np.pi)
        pl.ylabel('yaw (deg)')
        pl.xlabel('time (secs)')
        pl.grid('on')
        
        pl.subplot(337)
        pl.plot(t, alpha * 180./np.pi)
        pl.xlabel('time (seconds)')
        pl.ylabel('alpha (deg)')
        pl.grid('on')
    
        pl.subplot(338)
        pl.plot(t, beta * 180./np.pi)
        pl.xlabel('time (seconds)')
        pl.ylabel('beta (deg)')
        pl.grid('on')
    
        pl.subplot(339)
        pl.plot(t, gamma * 180./np.pi)
        pl.xlabel('time (seconds)')
        pl.ylabel('gamma (deg)')
        pl.grid('on')
                
def main():
    ax = a3.Axes3D(pl.figure(1))
    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(-100, 100)
    ax.set_zlim3d(0, 100)
    
    initial_state = [0, 0, 0, 0.0, 0., 0.0, 0, 0, 0, 0, 0, 0]
    tester = Tester(initial_state, 0, '../configs/aerosonde.yaml', ax, T_sim=15.0, dt = 1.0/200.0)
    tester.test_autopilot(72.0, 0, 500, animate = True)
    #tester.test_trim(35.0, 0.025, 100, 15000, animate = True)
    tester.plot()

        
if __name__ == '__main__':
    main()
