import numpy as np
from filters.ekf import EKF

class RollPitchEstimator(EKF):
    #u = p, q, r, Va
    #x = roll, pitch
    g = 9.81
    def f(self, u):
        roll = self.x[0]
        pitch = self.x[1]
        p = u[0]
        q = u[1]
        r = u[2]        
        sr = np.sin(roll)
        cr = np.cos(roll)
        tp = np.tan(pitch)
        _f = np.zeros((2,), dtype=np.double)
        _f[0] = p + q * sr * tp + r * cr * tp
        _f[1] = q * cr - r * sr
        return _f

    def h(self, u):
        roll = self.x[0]
        pitch = self.x[1]
        p = u[0]
        q = u[1]
        r = u[2]        
        Va = u[3]        
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cr = np.cos(roll)
        sr = np.sin(roll)
        _h = np.zeros((3,), dtype=np.double)
        _h[0] = q * Va * sp + self.g * sp
        _h[1] = r * Va * cp - p * Va * sp - self.g * cp * sr
        _h[2] = -q * Va * cp - self.g * cp * cr
        return _h

    def F(self, u):        
        roll = self.x[0]
        pitch = self.x[1]
        q = u[1]
        r = u[2]        
        sr = np.sin(roll)        
        cr = np.cos(roll)
        tp = np.tan(pitch)
        cp = np.cos(pitch)
        _dfdx = np.array([[q * cr * tp - r * sr * tp, (q * sr - r * cr)/cp**2], 
                          [-q * sr - r * cr, 0]], dtype = np.double) 
        return _dfdx

    def H(self, u):
        roll = self.x[0]
        pitch = self.x[1]
        p = u[0]
        q = u[1]
        r = u[2]        
        Va = u[3]        
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cr = np.cos(roll)
        sr = np.sin(roll)
        _dhdx = np.zeros((3, 2), dtype=np.double)    
        _dhdx[0, 0] = 0
        _dhdx[0, 1] = q * Va * cp + self.g * cp
        _dhdx[1, 0] = -self.g * cr * cp - r * Va * sp
        _dhdx[1, 1] = -p * Va * cp + self.g * sr * sp
        _dhdx[2, 0] = self.g * sr * cp
        _dhdx[2, 1] = (1 * Va + self.g * cr) * sp        
        return _dhdx
        
class PositionHeadingEstimator(EKF):
    #u = Va, q, r, phi, theta
    #x = pn, pe, Vg, chi, wn, we, psi
    g = 9.81
    def f(self, u):
        Vg = self.x[2]
        chi = self.x[3]
        wn = self.x[4]
        we = self.x[5]
        psi = self.x[6]        
        Va = u[0]        
        q = u[1]
        r = u[2]
        roll = u[3]
        pitch = u[4]
        sr = np.sin(roll)
        cr = np.cos(roll)
        cp = np.cos(pitch)
        psi_dot = q * sr/cp + r * cr/cp
        _f = np.zeros((7,), dtype=np.double)
        _f[0] = Vg * np.cos(chi)
        _f[1] = Vg * np.sin(chi)
        _f[2] = (Va * np.cos(psi) + wn) * -Va/Vg * psi_dot * np.sin(psi) 
        + (Va * np.sin(psi) + we)* Va/Vg * psi_dot * np.cos(psi)
        _f[3] = self.g/Vg * np.tan(roll) * np.cos(chi - psi)
        _f[4] = 0.0
        _f[5] = 0.0
        _f[6] = q * sr/cp + r * cr/cp
        return _f
    
    def h(self, u):
        pn = self.x[0]
        pe = self.x[1]
        Vg = self.x[2]
        chi = self.x[3]
        wn = self.x[4]
        we = self.x[5]
        psi = self.x[6]
        Va = u[0]
        _h = np.zeros((6,), dtype = np.double)
        _h[0] = pn
        _h[1] = pe
        _h[2] = Vg
        _h[3] = chi
        _h[4] = Va * np.cos(psi) + wn - Vg * np.cos(chi)
        _h[5] = Va * np.sin(psi) + we - Vg * np.sin(chi)        
    
    def F(self, u):
        Vg = self.x[2]
        chi = self.x[3]
        wn = self.x[4]
        we = self.x[5]
        psi = self.x[6]        
        Va = u[0]        
        q = u[1]
        r = u[2]
        roll = u[3]
        pitch = u[4]
        sr = np.sin(roll)
        cr = np.cos(roll)
        tr = np.tan(roll)
        cp = np.cos(pitch)
        psi_dot = q * sr/cp + r * cr/cp
        schi = np.sin(chi)
        cchi = np.cos(chi)
        spsi = np.sin(psi)
        cpsi = np.cos(psi)
        Vg_dot = 1/Vg * (Va * cpsi + wn) * (-Va * psi_dot * spsi) 
        + 1/Vg * (Va * spsi + we) * (Va * psi_dot * cpsi) 
        _F = np.zeros((7, 7), dtype = np.double)
        _F[0, 2] = cchi
        _F[0, 3] = -Vg * schi
        _F[1, 2] = schi
        _F[1, 3] = Vg * cchi
        _F[2, 2] = -Vg_dot/Vg
        _F[2, 4] = -psi_dot * Va * spsi
        _F[2, 5] = psi_dot * Va * cpsi
        _F[2, 6] = -psi_dot * Va/Vg * (wn * cpsi + we * spsi)
        _F[3, 2] = -self.g/Vg**2 * tr * np.cos(chi - psi)
        _F[3, 3] = self.g/Vg * tr * np.sin(chi - psi)
        _F[3, 6] = self.g/Vg * tr * np.sin(chi - psi)
        return _F
    
    def H(self, u):
        Vg = self.x[2]
        chi = self.x[3]        
        psi = self.x[6]
        schi = np.sin(chi)
        cchi = np.cos(chi)
        Va = u[0]   
        _H = np.zeros((6, 7), dtype = np.double)
        _H[0, 0] = 1
        _H[1, 1] = 1
        _H[2, 2] = 1
        _H[3, 3] = 1
        _H[4, 2] = -cchi
        _H[4, 3] = Vg * schi
        _H[4, 4] = 1
        _H[4, 6] = -Va * np.sin(psi)
        _H[5, 2] = -schi
        _H[5, 3] = Vg * cchi
        _H[5, 5] = 1
        _H[5, 6] = Va * np.cos(psi)
        return _H