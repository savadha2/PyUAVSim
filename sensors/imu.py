import numpy as np

class IMU:
    g = 9.81
    def __init__(self, sigma, manufacturer=None, model=None):
        self.manufacturer = manufacturer
        self.model = model
        self.sigma = sigma        
    
    def __call__(self, xtrue, force, mass):
        #assumption - bias is compensated for already
        a_true = np.zeros((3, ), dtype = np.double)
        a_measured = np.zeros((3, ), dtype = np.double)
        theta = xtrue[6]
        phi = xtrue[7]
        a_true[0] = force[0]/mass + self.g * np.sin(theta)
        a_true[1] = force[1]/mass - self.g * np.cos(theta) * np.sin(phi)
        a_true[2] = force[2]/mass - self.g * np.sco(theta) * np.cos(phi)
        a_measured = a_true + self.sigma * np.random.randn(3,)
        return a_measured
    