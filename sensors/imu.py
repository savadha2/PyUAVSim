import numpy as np

class IMU:
    def __init__(self, sigma, nsensors = 3, manufacturer=None, model=None):
        self.manufacturer = manufacturer
        self.model = model
        self.sigma = sigma
        self.nsensors = nsensors
    
    def __call__(self, a_truth):
        #assumption - bias is compensated for already
        a_measured = a_truth + self.sigma * np.random.randn(self.nsensors, 1)
        return a_measured
    