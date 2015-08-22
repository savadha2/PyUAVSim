# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:10:43 2015

@author: sharath
"""
from scipy.integrate import ode
import yaml
import errno
import numpy as np
from functools import partial

class DynamicsBase(object):
    def __init__(self, x0, t0, dt_integration = 1e-3,):
        self.integrator = None
        self.dt = dt_integration
        self.x0 = x0
        self.t0 = t0
        self.x = x0
        self.t = t0
    
    def set_integrator(self, ode_func, integrator, jac = None, **kwargs):
        self.integrator = ode(ode_func, jac).set_integrator(integrator, **kwargs)
        self.integrator.set_initial_value(self.x0, self.t0)
        
    def integrate(self, t1):
        if self.integrator is None:
            raise Exception('initialize integrator first using set_integrator')
        while self.integrator.successful() and self.integrator.t < t1:
            self.integrator.integrate(self.integrator.t + self.dt)
            self.x = self.integrator.y
            self.t = self.integrator.t

class  FixedWingUAVDynamics(DynamicsBase):
    def __init__(self, x0, t0, dt_integration, config_file):
        self.config = None
        try:
            with open(config_file) as f:
                self.config = yaml.load(f)
        except:
            raise IOError(errno.ENOENT, 'File not found', config_file)
        super(FixedWingUAVDynamics, self).__init__(x0, t0, dt_integration)
        self.t0 = t0
        self.set_integrator(FixedWingUAVDynamics.dynamics, 'dop853', jac = None, atol = 1e-6)        
        self.partial_forces_and_moments = partial(FixedWingUAVDynamics.forces_and_moments, config = self.config)
        self._control_inputs = None
        self.actuator_commands = [-0.083782251798531229, -5.2373899469808999e-07, 0.0, 0.50102375844986302]
        self.integrator.set_f_params(self.config, self.actuator_commands, self.partial_forces_and_moments)
    @staticmethod
    def forces_and_moments(y, control_inputs, config):
        mass = config['params']['mass']
        S = config['params']['S']
        b = config['params']['b']
        c = config['params']['c']
        rho = config['params']['rho']
        e = config['params']['e']
        S_prop = config['params']['S_prop']
        k_motor = config['params']['k_motor']
        kT_p = config['params']['kT_p']
        kOmega = config['params']['kOmega']
        
        Clong_coeffs = config['longitudinal_coeffs']
        Clateral_coeffs = config['lateral_coeffs']
        
        #pn = y[0]
        #pe = y[1]
        #pd = y[2]
        u = y[3]
        v = y[4]
        w = y[5]                
        phi = y[6]
        theta = y[7]
        psi = y[8]
        p = y[9]
        q = y[10]
        r = y[11]        

        Va = np.sqrt(u**2 + v**2 + w**2)
        alpha = np.arctan(w/u)
        beta = np.arcsin(v/Va)
                
        delta_e = control_inputs[0]
        delta_a = control_inputs[1]
        delta_r = control_inputs[2]
        delta_t = control_inputs[3]

        def longitudinal_aerodynamic_forces_moments():#srho, c, S, Va, Clong_coeffs, alpha, q, delta_e):
            CL0 = Clong_coeffs['CL0']
            CL_alpha = Clong_coeffs['CL_alpha']
            CL_q = Clong_coeffs['CL_q']
            CL_delta_e = Clong_coeffs['CL_delta_e']
            M = Clong_coeffs['M']
            alpha_0 = Clong_coeffs['alpha_0']
            c1 = np.exp(-M * (alpha - alpha_0))
            c2 = np.exp(M * (alpha + alpha_0))
            sigmoid_alpha = (1 + c1 + c2)/((1 + c1) * (1 + c2))
            CL_alpha_NL = (1. - sigmoid_alpha) * (CL0 + CL_alpha * alpha) + sigmoid_alpha * 2. * np.sign(alpha) * np.sin(alpha) * np.sin(alpha) * np.cos(alpha)
            lift = 0.5 * rho * Va**2 * S * (CL_alpha_NL + CL_q * c * q * 0.5/Va + CL_delta_e * delta_e)

            CD0 = Clong_coeffs['CD0']
            CD_alpha = Clong_coeffs['CD_alpha']
            CD_q = Clong_coeffs['CD_q']
            CD_delta_e = Clong_coeffs['CD_delta_e']
            CD_p = Clong_coeffs['CD_p']
            AR = b**2/S
            CD_alpha = CD_p + (CL0 + CL_alpha * alpha)**2/(np.pi * e * AR)
            drag = 0.5 * rho * Va**2 * S * (CD_alpha + CD_q * c * q * 0.5/Va + CD_delta_e * delta_e)
            
            Cm0 = Clong_coeffs['Cm0']
            Cm_alpha = Clong_coeffs['Cm_alpha']
            Cm_q = Clong_coeffs['Cm_q']
            Cm_delta_e = Clong_coeffs['Cm_delta_e']
            Cm_alpha = Cm0 + Cm_alpha * alpha
            #delta_e = -Cm_alpha/Cm_delta_e
            m = 0.5 * rho * Va**2 * S * c * (Cm_alpha + Cm_q * c * q * 0.5/Va + Cm_delta_e * delta_e)

            fx = -drag * np.cos(alpha) + lift * np.sin(alpha)
            fz = -drag * np.sin(alpha) - lift * np.cos(alpha)
            return fx, fz, m
            
        def lateral_forces_moments():#rho, b, S, Va, lateral_coeffs, beta, p, r, delta_a, delta_r):
            const = 0.5 * rho * Va**2 * S
            CY0 = Clateral_coeffs['CY0']
            CY_beta = Clateral_coeffs['CY_beta']
            CY_p = Clateral_coeffs['CY_p']
            CY_r = Clateral_coeffs['CY_r']
            CY_delta_a = Clateral_coeffs['CY_delta_a']
            CY_delta_r = Clateral_coeffs['CY_delta_r']
            fy = const * (CY0 + CY_beta * beta + CY_p * b * p * 0.5/Va + CY_r * r * b * 0.5/Va + CY_delta_a * delta_a + CY_delta_r * delta_r)
            
            Cl0 = Clateral_coeffs['Cl0']
            Cl_beta = Clateral_coeffs['Cl_beta']
            Cl_p = Clateral_coeffs['Cl_p']
            Cl_r = Clateral_coeffs['Cl_r']
            Cl_delta_a = Clateral_coeffs['Cl_delta_a']
            Cl_delta_r = Clateral_coeffs['Cl_delta_r']
            l = b * const * (Cl0 + Cl_beta * beta + Cl_p * b * p * 0.5/Va + Cl_r * r * b * 0.5/Va + Cl_delta_a * delta_a + Cl_delta_r * delta_r)

            Cn0 = Clateral_coeffs['Cn0']
            Cn_beta = Clateral_coeffs['Cn_beta']
            Cn_p = Clateral_coeffs['Cn_p']
            Cn_r = Clateral_coeffs['Cn_r']
            Cn_delta_a = Clateral_coeffs['Cn_delta_a']
            Cn_delta_r = Clateral_coeffs['Cn_delta_r']
            n = b * const * (Cn0 + Cn_beta * beta + Cn_p * b * p * 0.5/Va + Cn_r * r * b * 0.5/Va + Cn_delta_a * delta_a + Cn_delta_r * delta_r)
            return fy, l, n
            
        def gravitational_forces():#mass, theta, phi):
            g = 9.81
            fx = -mass * g * np.sin(theta)
            fy = mass * g * np.cos(theta) * np.sin(phi)
            fz = mass * g * np.cos(theta) * np.cos(phi)
            return fx, fy, fz
        
        def propeller_forces():#rho, S_prop, Clong_coeffs, k_motor, delta_t = 1):
            C_prop = Clong_coeffs['C_prop']
            fx = 0.5 * rho * S_prop * C_prop * (k_motor**2 * delta_t**2 - Va**2)
            fy = 0.
            fz = 0.
            return fx, fy, fz            
        def propeller_torques():#kT_p, kOmega, delta_t = 1):
            l = -kT_p * kOmega **2 * delta_t **2
            m = 0.
            n = 0.
            return l, m, n
            
        def wind(Va):
            return 0., 0., 0.
                      
        f_lon_aerodynamic_x, f_lon_aerodynamic_z, m_aerodynamic = longitudinal_aerodynamic_forces_moments()#rho, c, b, S, e, Va, Clong_coeffs, alpha, q, delta_e)
        f_lat_aerodynamic_y, l_aerodynamic, n_aerodynamic = lateral_forces_moments()#rho, b, S, Va, Clateral_coeffs, beta, p, r, delta_a, delta_r)
        f_gravity_x, f_gravity_y, f_gravity_z = gravitational_forces()#mass, theta, phi)
        f_prop_x, f_prop_y, f_prop_z = propeller_forces()#rho, S_prop, Clong_coeffs, k_motor, delta_t)
        l_prop, m_prop, n_prop = propeller_torques()#kT_p, kOmega, delta_t)
        fx = f_lon_aerodynamic_x + f_gravity_x + f_prop_x
        fy = f_lat_aerodynamic_y + f_gravity_y + f_prop_y
        fz = f_lon_aerodynamic_z + f_gravity_z + f_prop_z
        l = l_aerodynamic + l_prop        
        m = m_aerodynamic + m_prop        
        n = n_aerodynamic + n_prop
        #print 'fz: ', fz
        return [fx, fy, fz], [l, m, n]
        
    @staticmethod
    def dynamics(t, y, config, control_inputs, forces_and_moments):
        mass = config['params']['mass']
        Jx = config['params']['Jx']
        Jy = config['params']['Jy']
        Jz = config['params']['Jz']
        Jxz = config['params']['Jxz']
        gamma_0 = Jx * Jz - Jxz**2
        gamma_1 = (Jxz * (Jx - Jy + Jz))/gamma_0
        gamma_2 = (Jz * (Jz - Jy) + Jxz**2)/gamma_0
        gamma_3 = Jz/gamma_0
        gamma_4 = Jxz/gamma_0
        gamma_5 = (Jz - Jx)/Jy
        gamma_6 = Jxz/Jy
        gamma_7 = ((Jx - Jy)* Jx + Jxz**2)/gamma_0
        gamma_8 = Jx/gamma_0
                
        #pn = y[0]
        #pe = y[1]
        #pd = y[2]
        u = y[3]
        v = y[4]
        w = y[5]                
        phi = y[6]
        theta = y[7]
        psi = y[8]
        p = y[9]
        q = y[10]
        r = y[11]        
        cr = np.cos(phi)
        sr = np.sin(phi)
        
        cp = np.cos(theta)
        sp = np.sin(theta)
        tp = np.tan(theta)
        
        cy = np.cos(psi)
        sy = np.sin(psi)
        
        forces, moments = forces_and_moments(y, control_inputs)
        fx = forces[0]
        fy = forces[1]
        fz = forces[2]
        l = moments[0]
        m = moments[1]
        n = moments[2]
        
        dy = np.zeros((12,), dtype = np.double)  
        dy[0] = cp * cy * u + (sr * sp * cy - cr * sy) * v + (cr * sp * cy + sr * sy) * w
        dy[1] = cp * sy * u + (sr * sp * sy + cr * cy) * v + (cr * sp * sy - sr * cy) * w
        dy[2] = -sp * u + sr * cp * v + cr * cp * w
        dy[3] = r * v - q * w + fx/mass
        dy[4] = p * w - r * u + fy/mass
        dy[5] = q * u - p * v + fz/mass
        dy[6] = p + sr * tp * q + cr * tp * r
        dy[7] = cr * q - sr * r
        dy[8] = sr/cp * q + cr/cp * r        
        dy[9] = gamma_1 * p * q - gamma_2 * q * r + gamma_3 * l + gamma_4 * n
        dy[10] = gamma_5 * p * r - gamma_6 * (p * p - r * r) + m/Jy
        #print 'theta: ', theta
        dy[11] = gamma_7 * p * q - gamma_1 * q * r + gamma_4 * l + gamma_8 * n
                
        return dy
    
    def compute_trimmed_states_inputs(self, Va, gamma, turn_radius, alpha, beta, phi):
        config = self.config
        R = turn_radius
        g = 9.81
        mass = config['params']['mass']
        Jx = config['params']['Jx']
        Jy = config['params']['Jy']
        Jz = config['params']['Jz']
        Jxz = config['params']['Jxz']
        gamma_0 = Jx * Jz - Jxz**2
        gamma_1 = (Jxz * (Jx - Jy + Jz))/gamma_0
        gamma_2 = (Jz * (Jz - Jy) + Jxz**2)/gamma_0
        gamma_3 = Jz/gamma_0
        gamma_4 = Jxz/gamma_0
        #gamma_5 = (Jz - Jx)/Jy
        #gamma_6 = Jxz/Jy
        gamma_7 = ((Jx - Jy)* Jx + Jxz**2)/gamma_0
        gamma_8 = Jx/gamma_0
        S = config['params']['S']
        b = config['params']['b']
        c = config['params']['c']
        rho = config['params']['rho']
        e = config['params']['e']
        S_prop = config['params']['S_prop']
        k_motor = config['params']['k_motor']
        #kT_p = config['params']['kT_p']
        #kOmega = config['params']['kOmega']
        Clong_coeffs = config['longitudinal_coeffs']
        Clateral_coeffs = config['lateral_coeffs']
        
        x = np.zeros((12,), dtype = np.double)    
        x[3] = Va * np.cos(alpha) * np.cos(beta)
        x[4] = Va * np.sin(beta)
        x[5] = Va * np.sin(alpha) * np.cos(beta)
        theta = alpha + gamma
        x[6] = phi
        x[7] = theta
        x[9] = -Va/R * np.sin(theta)
        x[10] = Va/R * np.sin(phi) * np.cos(theta)
        x[11] = Va/R * np.cos(phi) * np.cos(theta)        
        #u = x[3]
        v = x[4]
        w = x[5]
        p = x[9]
        q = x[10]
        r = x[11]
        
        C0 = 0.5 * rho * Va**2 * S
        
        def delta_e():
            C1 = (Jxz * (p**2 - r**2) + (Jx - Jz) * p *r)/(C0 * c)
            Cm0 = Clong_coeffs['Cm0']
            Cm_alpha = Clong_coeffs['Cm_alpha']
            Cm_q = Clong_coeffs['Cm_q']
            Cm_delta_e = Clong_coeffs['Cm_delta_e']
            Cm_alpha = Cm0 + Cm_alpha * alpha
            return (C1 - Cm_alpha - Cm_q * c * q * 0.5/Va)/Cm_delta_e
        delta_e = delta_e()
        
        def delta_t():
            CL0 = Clong_coeffs['CL0']
            CL_alpha = Clong_coeffs['CL_alpha']
            M = Clong_coeffs['M']
            alpha_0 = Clong_coeffs['alpha_0']
            CD_alpha = Clong_coeffs['CD_alpha']
            CD_p = Clong_coeffs['CD_p']
            CD_q = Clong_coeffs['CD_q']
            CL_q = Clong_coeffs['CL_q']
            CL_delta_e = Clong_coeffs['CL_delta_e']
            CD_delta_e = Clong_coeffs['CD_delta_e']
            C_prop = Clong_coeffs['C_prop']
            c1 = np.exp(-M * (alpha - alpha_0))
            c2 = np.exp(M * (alpha + alpha_0))
            sigmoid_alpha = (1 + c1 + c2)/((1 + c1) * (1 + c2))
            CL_alpha_NL = (1. - sigmoid_alpha) * (CL0 + CL_alpha * alpha) + sigmoid_alpha * 2. * np.sign(alpha) * np.sin(alpha) * np.sin(alpha) * np.cos(alpha)
            AR = b**2/S
            CD_alpha = CD_p + (CL0 + CL_alpha * alpha)**2/(np.pi * e * AR)
            CX  = -CD_alpha * np.cos(alpha) + CL_alpha_NL * np.sin(alpha)
            CX_delta_e = -CD_delta_e * np.cos(alpha) + CL_delta_e * np.sin(alpha)
            CX_q = -CD_q * np.cos(alpha) + CL_q * np.sin(alpha)
            C2 = 2 * mass * (-r * v  + q * w + g * np.sin(theta))
            C3 = -2 * C0 * (CX + CX_q * c * q * 0.5/Va + CX_delta_e * delta_e )
            C4 = rho * C_prop * S_prop * k_motor**2
            print 'C2, C3, C4', C2, C3, C4
            return np.sqrt((C2 + C3)/C4 + Va**2/k_motor**2)
        delta_t = delta_t()
        
        def delta_a_delta_r():
            Cl_delta_a = Clateral_coeffs['Cl_delta_a']
            Cn_delta_a = Clateral_coeffs['Cn_delta_a']
            Cl_delta_r = Clateral_coeffs['Cl_delta_r']
            Cn_delta_r = Clateral_coeffs['Cn_delta_r']
            Cl0 = Clateral_coeffs['Cl0']
            Cn0 = Clateral_coeffs['Cn0']
            Cl_p = Clateral_coeffs['Cl_p']
            Cn_p = Clateral_coeffs['Cn_p']
            Cl_beta = Clateral_coeffs['Cl_beta']
            Cn_beta = Clateral_coeffs['Cn_beta']
            Cl_r = Clateral_coeffs['Cl_r']
            Cn_r = Clateral_coeffs['Cn_r']
            Cp_delta_a = gamma_3 * Cl_delta_a + gamma_4 * Cn_delta_a
            Cp_delta_r = gamma_3 * Cl_delta_r + gamma_4 * Cn_delta_r
            Cr_delta_a = gamma_4 * Cl_delta_a + gamma_8 * Cn_delta_a
            Cr_delta_r = gamma_4 * Cl_delta_r + gamma_8 * Cn_delta_r
            Cp_0 = gamma_3 * Cl0 + gamma_4 * Cn0
            Cp_beta = gamma_3 * Cl_beta + gamma_4 * Cn_beta
            Cp_p = gamma_3 * Cl_p + gamma_4 * Cn_p
            Cp_r = gamma_3 * Cl_r + gamma_4 * Cn_r
            Cr_0 = gamma_4 * Cl0 + gamma_8 * Cn0
            Cr_beta = gamma_4 * Cl_beta + gamma_8 * Cn_beta
            Cr_p = gamma_4 * Cl_p + gamma_8 * Cn_p
            Cr_r = gamma_4 * Cl_r + gamma_8 * Cn_r

            
            C5 = (-gamma_1 * p * q + gamma_2 * q * r)/(C0 * b)
            C6 = (-gamma_7 * p * q + gamma_1 * q * r)/(C0 * b)
            v0 = C5 - Cp_0 - Cp_beta * beta - Cp_p * b * p * 0.5/Va - Cp_r * b * r * 0.5/Va
            v1 = C6 - Cr_0 - Cr_beta * beta - Cr_p * b * p * 0.5/Va - Cr_r * b * r * 0.5/Va
            v = [v0, v1]
            B = np.array([[Cp_delta_a, Cp_delta_r], [Cr_delta_a, Cr_delta_r]], dtype = np.double)
            if Cp_delta_r == 0. and Cr_delta_r == 0.:
                return [v0/B[0][0], 0.]
            elif Cp_delta_a == 0. and Cr_delta_a == 0.:
                return [0.0, v1/B[1][1]]
            else:
                _delta_a_delta_r = np.dot(np.linalg.inv(B), v)
                return _delta_a_delta_r[0], _delta_a_delta_r[1]
            
        delta_a, delta_r = delta_a_delta_r()

        control_inputs = [delta_e, delta_a, delta_r, delta_t]
        
        return x, control_inputs
    
    def J(self, alpha, beta, phi, Va, gamma, turn_radius, config):
        mass = config['params']['mass']
        Jx = config['params']['Jx']
        Jy = config['params']['Jy']
        Jz = config['params']['Jz']
        Jxz = config['params']['Jxz']
        gamma_0 = Jx * Jz - Jxz**2
        gamma_1 = (Jxz * (Jx - Jy + Jz))/gamma_0
        gamma_2 = (Jz * (Jz - Jy) + Jxz**2)/gamma_0
        gamma_3 = Jz/gamma_0
        gamma_4 = Jxz/gamma_0
        gamma_5 = (Jz - Jx)/Jy
        gamma_6 = Jxz/Jy
        gamma_7 = ((Jx - Jy)* Jx + Jxz**2)/gamma_0
        gamma_8 = Jx/gamma_0
        
        trimmed_state, trimmed_control = self.compute_trimmed_states_inputs(Va, gamma, turn_radius, alpha, beta, phi)
        forces, moments = self.partial_forces_and_moments(trimmed_state, trimmed_control)
        fx = forces[0]
        fy = forces[1]
        fz = forces[2]
        l = moments[0]
        m = moments[1]
        n = moments[2]

        y = trimmed_state
        u = y[3]
        v = y[4]
        w = y[5]                
        phi = y[6]
        theta = y[7]
        p = y[9]
        q = y[10]
        r = y[11]        
        cr = np.cos(phi)
        sr = np.sin(phi)       
        cp = np.cos(theta)
        sp = np.sin(theta)
        tp = np.tan(theta)
        
        f = np.zeros((12,), dtype = np.double)  
        f[2] = -sp * u + sr * cp * v + cr * cp * w
        f[3] = r * v - q * w + fx/mass
        f[4] = p * w - r * u + fy/mass
        f[5] = q * u - p * v + fz/mass
        f[6] = p + sr * tp * q + cr * tp * r
        f[7] = cr * q - sr * r
        f[8] = sr/cp * q + cr/cp * r        
        f[9] = gamma_1 * p * q - gamma_2 * q * r + gamma_3 * l + gamma_4 * n
        f[10] = gamma_5 * p * r - gamma_6 * (p * p - r * r) + m/Jy
        f[11] = gamma_7 * p * q - gamma_1 * q * r + gamma_4 * l + gamma_8 * n
        
        xdot = np.zeros((12,), dtype = np.double)
        xdot[2] = -Va * np.sin(gamma)
        xdot[8] = Va/turn_radius * np.cos(gamma)
        J = np.linalg.norm(xdot[2:] - f[2:])**2
        print 'xdot: ', xdot
        print 'trimmed control: ', trimmed_control
        print 'f: ', f
        print 'J: ', J
        print 'f[10]: ', f[10]
        return J
            
    def trim(self, Va, gamma, turn_radius):
        R = turn_radius
        J = partial(self.J, Va = Va, gamma = gamma, turn_radius = R, config = self.config)
        def gradient_descent(alpha, beta, phi, kappa = 1e-6, max_iters = 500, epsilon = 1e-4):
            iters = 0
            J0 = np.inf
            while(iters < max_iters):
                alpha_plus = alpha + epsilon
                beta_plus = beta + epsilon
                phi_plus = phi + epsilon
                print '----------START------------'
                J0 = J(alpha, beta, phi)
                dJ_dalpha = (J(alpha_plus, beta, phi) - J0)/epsilon
                print '-----------END--------------'
                dJ_dbeta = (J(alpha, beta_plus, phi) - J0)/epsilon
                dJ_dphi = (J(alpha, beta, phi_plus) - J0)/epsilon
                alpha = alpha - kappa * dJ_dalpha
                beta = beta - kappa * dJ_dbeta
                phi = phi - kappa * dJ_dphi
                iters += 1
                print 'J0, dJ_alpha, dJ_beta, dJ_phi', J0, dJ_dalpha, dJ_dbeta, dJ_dphi
            return alpha, beta, phi
#        alpha = -0.0
#        beta = 0.
#        phi = 0.
        
        alpha, beta, phi = gradient_descent(0., 0., 0.)
        print 'alpha, beta, phi: ', alpha, beta, phi
        trimmed_state, trimmed_control = self.compute_trimmed_states_inputs(Va, gamma, R, alpha, beta, phi)
        
        return trimmed_state, trimmed_control
    
    @property
    def control_inputs(self):
        return self._control_inputs
    @control_inputs.setter
    def control_inputs(self, inputs):
        self._control_inputs = inputs
        self.integrator.set_f_params(self.config, self._control_inputs, self.partial_forces_and_moments)
    def test_compute_alpha(self, y, config):
        actuator_commands = self.actuator_commands
        mass = config['params']['mass']
        S = config['params']['S']
        b = config['params']['b']
        c = config['params']['c']
        rho = config['params']['rho']
        
        Clong_coeffs = config['longitudinal_coeffs']
        
        #pn = y[0]
        #pe = y[1]
        #pd = y[2]
        u = y[3]
        v = y[4]
        w = y[5]                
        phi = y[6]
        theta = y[7]
        psi = y[8]
        p = y[9]
        q = y[10]
        r = y[11]        

        Va = np.sqrt(u**2 + v**2 + w**2)
#        alpha = np.arctan(w/u)
#        beta = np.arcsin(v/Va)
                
        delta_e = actuator_commands[0]
        
        g = 9.81
        fz_gravity = mass * g * np.cos(theta) * np.cos(phi)
        
        CL0 = Clong_coeffs['CL0']
        CL_alpha = Clong_coeffs['CL_alpha']
        CL_q = Clong_coeffs['CL_q']
        CL_delta_e = Clong_coeffs['CL_delta_e']        
        
        alpha = (fz_gravity/(0.5 * rho * Va**2 * S) - CL_q * c * q * 0.5/Va - CL_delta_e * delta_e - CL0)/(CL_alpha)
        
        return alpha
        
        
        
            
            
            
            
            
            
            
            
            
            