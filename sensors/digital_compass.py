import numpy as np

class DigitalCompass:    
    def __init__(self, bias, sigma, manufacturer=None, model=None):
        self.bias = bias
        self.sigma = sigma
        self.manufacturer = manufacturer
        self.model = model
    
    def __call__(self, heading_true):        
        heading_measured = heading_true+ self.bias + self.sigma * np.random.randn(1,)
        return heading_measured