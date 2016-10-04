# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 13:45:06 2016

@author: sharath
"""

import numpy as np

#TODO: make filletbased path maanager into a separate class and have a factory
class FixedWingUAVPathManager(object):
    def __init__(self):
        self.waypoints = []
        self.fsm_state = 1
        self.reached_wp = False

    def __call__(self, waypoints, manager='fillet'):
        if manager=='fillet':
            flag, r, q, c, rho, lam = self.filletpathmanager(waypoints)
        else:
            raise Exception('unsupported path manager type: ', manager)
        return self.reached_wp, flag, r, q, c, rho, lam

    def filletpathmanager(self, waypoints, fillet_radius = 70):
        if waypoints.shape[0] < 3:
            self.ua.message("No more waypoints. Will stay on course ! ")
            return -1, None, None, None, None, None
        location = self.ua.location
        q = np.zeros((3, 2), dtype = np.double)
        q_f = np.zeros((3, ), dtype = np.double)
        r = np.zeros((3, ), dtype = np.double)
        c = np.zeros((3, ), dtype = np.double)
        rho = np.inf
        lam = 1.

        R = fillet_radius
        q[:, 0] = (waypoints[1] - waypoints[0])/np.linalg.norm(waypoints[1] - waypoints[0])
        q[:, 1] = (waypoints[2] - waypoints[1])/np.linalg.norm(waypoints[2] - waypoints[1])
        angle = np.arccos(np.dot(-q[:, 0], q[:, 1]))
        if self.fsm_state == 1:
            self.reached_wp = False
            flag = 1
            r = waypoints[0]
            q_f = q[:, 0]
            z = waypoints[1] - (R/np.tan(angle/2)) * q[:, 0]
            if self.in_plane(location[0:3], z, q[:, 0]):
                self.fsm_state = 2
        if self.fsm_state == 2:
            flag = 2
            delta_q = q[:, 0] - q[:, 1]
            norm_delta_q = np.linalg.norm(delta_q)
            if norm_delta_q > 1e-6:
                c = waypoints[1] - (R/np.tan(angle/2)) * (delta_q)/norm_delta_q
            else:
                c = waypoints[1]

            rho = R
            lam = np.sign(q[0, 0] * q[1, 1] - q[1, 0] * q[0, 1])
            z = waypoints[1] + (R/np.tan(angle/2)) * q[:, 1]
            if self.in_plane(location[0:3], z, q[:, 1]):
                self.reached_wp = True
                self.fsm_state = 1

        return flag, r, q_f, c, rho, lam

    def in_plane(self, p, p0, p1, threshold = 25.0):
        x1 = p[0]; x2 = p0[0]; x3 = p1[0]
        y1 = p[1]; y2 = p0[1]; y3 = p1[1]
        z1 = -p[2]; z2 = p0[2]; z3 = p1[2]
        _in_plane  = (x1 - x2) * x3 + (y1 - y2) * y3 + (z1 - z2) * z3
        if abs(_in_plane) < threshold:
            return True
        else:
            return False