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
    def f(self, u):
        pass
    def h(self, u):
        pass
    def F(self, u):
        pass
    def H(self, u):
        pass    