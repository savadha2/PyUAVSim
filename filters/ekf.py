import numpy as np
from abc import ABCMeta, abstractmethod
class EKF(object):
    __metaclass__ = ABCMeta
    def __init__(self, x, P, Q, R, N):
        self.x = x
        self.P = P
        self.Q = Q
        self.R = R
        self.I = np.eye(len(x))
        #N is used for integrating f, dt/N. see __call__
        self.N = N
    
    @abstractmethod
    def f(self, u):pass        
    
    @abstractmethod
    def F(self, u):pass
    
    @abstractmethod    
    def h(self, u):pass
    
    @abstractmethod    
    def H(self, u):pass        
    
    def __call__(self, dt, u, measurements=None):
        dt_integ = dt/self.N
        for k in range(self.N):
            self.x += self.f(u) * dt_integ
            F = self.F(u)
            self.P += dt_integ(F * self.P * F.T + self.Q)
        
        if measurements is not None:
            H = self.H(u)
            PHT = self.P * H.T        
            S = H * PHT + self.R
            K = PHT * np.linalg.inv(S)
            y = measurements - self.h(u)            
            self.x = self.x + K * y
            self.P = (self.I - K * H) * self.P
        
        