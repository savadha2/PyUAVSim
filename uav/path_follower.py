# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:12:21 2016

@author: sharath
"""
import numpy as np

class FixedWingUAVPathFollower(object):
    def __init__(self):
        #@TODO: make the params below configurable
        self.k_path = 0.5
        self.k_orbit = 0.5
        self.chi_inf = np.pi/4.

    def straightline_follower(self, r, q, uav_state):
        p = uav_state[0:3]
        chi = uav_state[8]
        qn = q[0]
        qe = [1]
        qd = q[2]
        e_p= p - r
        def compute_altitude():
            rd = r[2]
            n = np.cross(q, np.array([0, 0, 1], dtype = np.double))
            n = n/np.linalg.norm(n)
            s = e_p - np.dot(e_p, n) * n
            h_c = -rd - np.linalg.norm(s[0:2]) * qd * 1.0/np.linalg.norm(q[0:2])
            return h_c

        chi_q = np.arctan2(qe, qn)
        while chi_q - chi < -np.pi:
            chi_q = chi_q + 2. * np.pi

        while chi_q - chi > np.pi:
            chi_q = chi_q - 2. * np.pi

        def compute_course_angle():
            cchi = np.cos(chi_q)
            schi = np.sin(chi_q)
            e_py = schi * e_p[0] + cchi * e_p[1]
            chi_c = chi_q - self.chi_inf * 2.0 * np.arctan(self.k_path * e_py)/np.pi
            return chi_c

        return compute_altitude(), compute_course_angle()

    def orbit_follower(self, c, rho, lam, uav_state):
        p = uav_state[0:3]
        chi = uav_state[8]
        h_c = -c[2]
        d = np.linalg.norm(c-p)
        psi = np.arctan2(c[1] - p[1], c[0] - p[0])
        while psi - chi < -np.pi:
            psi = psi + np.pi * 2.
        while psi - chi > np.pi:
            psi = psi - np.pi * 2.
        chi_c = psi + lam * (np.pi/2. + np.arctan(self.k_orbit * (d/rho-1.)))
        return h_c, chi_c

    def __call__(self, flag, r, q, c, rho, lam, uav_state):
        if flag == 1:
            h_c, chi_c = self.straightline_follower(r, q, uav_state)
        elif flag == 2:
            h_c, chi_c = self.orbit_follower(c, rho, lam, uav_state)
        return h_c, chi_c