import numpy as np

class Gyro:
    def __init__(self, sigma, nsensors = 3, manufacturer=None, model=None):
        self.manufacturer = manufacturer
        self.model = model
        self.sigma = sigma
        self.nsensors = nsensors
    
    def __call__(self, omega_truth):
        #assumption - bias is compensated for already
        omega_measured = omega_truth + self.sigma * np.random.randn(self.nsensors, 1)
        return omega_measured