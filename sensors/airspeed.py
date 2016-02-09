import numpy as np

class Airspeed:
    M = 0.0289644
    R = 8.31432
    P0 = 101325.0
    T0 = 288.15
    L0 = -0.0065
    g = 9.81
    def __init__(self, bias, sigma, manufacturer=None, model=None):
        self.bias = bias
        self.sigma = sigma
        self.manufacturer = manufacturer
        self.model = model
    
    def __call__(self, Va_true, temperature_f, h_asl):        
        T = 5.0/9 * (temperature_f - 32.0) + 273.15
        P = self.P0 * (self.T0/(self.T0 + self.L0 * h_asl)) ** (self.g * self.M * 1/self.R * 1/self.L0)
        rho = P * self.M/(self.R * T)
        diff_pressure_measured = rho * Va_true ** 2 * 0.5 + self.bias + self.sigma * np.random.randn(1,)
        return diff_pressure_measured