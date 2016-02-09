import numpy as np

class GPS:
    def __init__(self, k_gps, Ts, sigma_pos, sigma_horizontal_speed, manufacturer=None, model=None):
        self.manufacturer = manufacturer
        self.model = model
        self.sigma_pos = sigma_pos
        self.pos_err = np.zeros((3,), dtype=np.double)
        self.k_gps = k_gps
        self.Ts = Ts
        self.sigma_horizontal_speed = sigma_horizontal_speed
        
    #x = [pos, velocity]
    def __call__(self, x_true, va_true, heading_true, wind_true_ned):
        position_true = x_true
        self.pos_err = np.exp(-self.k_gps * self.Ts) * self.pos_err + self.sigma_pos * np.random.randn(3,)
        position_measured = position_true + self.pos_err
        x_measured = np.zeros(x_true.shape)
        x_measured[0:3] = position_measured
        
        vg_north = va_true * np.cos(heading_true) + wind_true_ned[0]
        vg_east = va_true * np.cos(heading_true) + wind_true_ned[1]
        vg_horizontal_true = np.sqrt(vg_north ** 2 + vg_east ** 2)
        if self.sigma_horizontal_speed.shape[0] == 1:
            sigma_vg_north = self.sigma_horizontal_speed[0]
            sigma_vg_east = self.sigma_horizontal_speed[0]
        else:
            sigma_vg_north = self.sigma_horizontal_speed[0]
            sigma_vg_east = self.sigma_horizontal_speed[1]
            
        sigma_v = np.sqrt((vg_north**2 * sigma_vg_north**2 + vg_east**2 * sigma_vg_east**2))/vg_horizontal_true
        vg_horizontal_measured = vg_horizontal_true + sigma_v * np.random.randn(1,)
        
        sigma_heading = np.sqrt((vg_north**2 * sigma_vg_east**2 + vg_east**2 * sigma_vg_north**2))/vg_horizontal_true
        heading_measured = np.arctan2(vg_east, vg_north) + sigma_heading * np.random.randn(1,)
        return x_measured, vg_horizontal_measured, heading_measured
    