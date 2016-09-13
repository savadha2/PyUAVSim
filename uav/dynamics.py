# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:10:43 2015

@author: sharath
"""
from scipy.integrate import ode
import numpy as np
from functools import partial

class DynamicsBase(object):
    def __init__(self, x0, t0, dt_integration = 1e-3):
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
    def __init__(self, x0, t0, dt_integration, attrs):
        self.attrs = attrs
        super(FixedWingUAVDynamics, self).__init__(x0, t0, dt_integration)
        self.t0 = t0
        self.set_integrator(FixedWingUAVDynamics.dynamics, 'dop853', jac = None, rtol = 1e-8)        
        self.partial_forces_and_moments = partial(FixedWingUAVDynamics.forces_and_moments, attrs = self.attrs)
        self._control_inputs = [0., 0., 0., 0.]
    @staticmethod
    def forces_and_moments(y, control_inputs, attrs):
        mass = attrs['params']['mass']
        S = attrs['params']['S']
        b = attrs['params']['b']
        c = attrs['params']['c']
        rho = attrs['params']['rho']
        e = attrs['params']['e']
        S_prop = attrs['params']['S_prop']
        k_motor = attrs['params']['k_motor']
        kT_p = attrs['params']['kT_p']
        kOmega = attrs['params']['kOmega']
        
        Clong_coeffs = attrs['longitudinal_coeffs']
        Clateral_coeffs = attrs['lateral_coeffs']
        
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
        if u > 0:
            alpha = np.arctan(w/u)
        elif Va == 0:
            alpha = 0
        else:
            alpha = np.pi/2

        if Va > 0:
            beta = np.arcsin(v/Va)
        else:
            beta = 0
                
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
            lift = 0.5 * rho * S * (CL_alpha_NL * Va**2 + CL_q * c * q * 0.5 * Va + CL_delta_e * delta_e * Va**2)

            CD0 = Clong_coeffs['CD0']
            CD_alpha = Clong_coeffs['CD_alpha']
            CD_q = Clong_coeffs['CD_q']
            CD_delta_e = Clong_coeffs['CD_delta_e']
            CD_p = Clong_coeffs['CD_p']
            AR = b**2/S
            CD_alpha = CD_p + (CL0 + CL_alpha * alpha)**2/(np.pi * e * AR)
            drag = 0.5 * rho * S * (CD_alpha * Va**2 + CD_q * c * q * 0.5 * Va + CD_delta_e * delta_e * Va**2)
            
            Cm0 = Clong_coeffs['Cm0']
            Cm_alpha = Clong_coeffs['Cm_alpha']
            Cm_q = Clong_coeffs['Cm_q']
            Cm_delta_e = Clong_coeffs['Cm_delta_e']
            Cm_alpha = Cm0 + Cm_alpha * alpha
            #delta_e = -Cm_alpha/Cm_delta_e
            m = 0.5 * rho * S * c * (Cm_alpha * Va**2 + Cm_q * c * q * 0.5 * Va + Cm_delta_e * delta_e * Va**2)

            fx = -drag * np.cos(alpha) + lift * np.sin(alpha)
            fz = -drag * np.sin(alpha) - lift * np.cos(alpha)
            return fx, fz, m
            
        def lateral_forces_moments():#rho, b, S, Va, lateral_coeffs, beta, p, r, delta_a, delta_r):
            const = 0.5 * rho * S
            CY0 = Clateral_coeffs['CY0']
            CY_beta = Clateral_coeffs['CY_beta']
            CY_p = Clateral_coeffs['CY_p']
            CY_r = Clateral_coeffs['CY_r']
            CY_delta_a = Clateral_coeffs['CY_delta_a']
            CY_delta_r = Clateral_coeffs['CY_delta_r']
            fy = const * (CY0 * Va**2 + CY_beta * beta * Va**2 + CY_p * b * p * 0.5 * Va + CY_r * r * b * 0.5 * Va + CY_delta_a * delta_a * Va**2 + CY_delta_r * delta_r * Va**2)
            
            Cl0 = Clateral_coeffs['Cl0']
            Cl_beta = Clateral_coeffs['Cl_beta']
            Cl_p = Clateral_coeffs['Cl_p']
            Cl_r = Clateral_coeffs['Cl_r']
            Cl_delta_a = Clateral_coeffs['Cl_delta_a']
            Cl_delta_r = Clateral_coeffs['Cl_delta_r']
            l = b * const * (Cl0 * Va**2 + Cl_beta * beta * Va**2 + Cl_p * b * p * 0.5 * Va + Cl_r * r * b * 0.5 * Va + Cl_delta_a * delta_a * Va**2 + Cl_delta_r * delta_r * Va**2)

            Cn0 = Clateral_coeffs['Cn0']
            Cn_beta = Clateral_coeffs['Cn_beta']
            Cn_p = Clateral_coeffs['Cn_p']
            Cn_r = Clateral_coeffs['Cn_r']
            Cn_delta_a = Clateral_coeffs['Cn_delta_a']
            Cn_delta_r = Clateral_coeffs['Cn_delta_r']
            n = b * const * (Cn0 * Va**2 + Cn_beta * beta * Va**2 + Cn_p * b * p * 0.5 * Va + Cn_r * r * b * 0.5 * Va + Cn_delta_a * delta_a * Va**2 + Cn_delta_r * delta_r * Va**2)
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
        return [fx, fy, fz], [l, m, n]
        
    @staticmethod
    def dynamics(t, y, attrs, control_inputs, forces_and_moments):
        mass = attrs['params']['mass']
        Jx = attrs['params']['Jx']
        Jy = attrs['params']['Jy']
        Jz = attrs['params']['Jz']
        Jxz = attrs['params']['Jxz']
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
        #print fx/mass
        dy[3] = r * v - q * w + fx/mass
        dy[4] = p * w - r * u + fy/mass
        dy[5] = q * u - p * v + fz/mass
        dy[6] = p + sr * tp * q + cr * tp * r
        dy[7] = cr * q - sr * r
        dy[8] = sr/cp * q + cr/cp * r        
        dy[9] = gamma_1 * p * q - gamma_2 * q * r + gamma_3 * l + gamma_4 * n
        dy[10] = gamma_5 * p * r - gamma_6 * (p * p - r * r) + m/Jy
        dy[11] = gamma_7 * p * q - gamma_1 * q * r + gamma_4 * l + gamma_8 * n
                
        return dy
    
    def compute_trimmed_states_inputs(self, Va, gamma, turn_radius, alpha, beta, phi):
        attrs = self.attrs
        R = turn_radius
        g = 9.81
        mass = attrs['params']['mass']
        Jx = attrs['params']['Jx']
        Jy = attrs['params']['Jy']
        Jz = attrs['params']['Jz']
        Jxz = attrs['params']['Jxz']
        gamma_0 = Jx * Jz - Jxz**2
        gamma_1 = (Jxz * (Jx - Jy + Jz))/gamma_0
        gamma_2 = (Jz * (Jz - Jy) + Jxz**2)/gamma_0
        gamma_3 = Jz/gamma_0
        gamma_4 = Jxz/gamma_0
        #gamma_5 = (Jz - Jx)/Jy
        #gamma_6 = Jxz/Jy
        gamma_7 = ((Jx - Jy)* Jx + Jxz**2)/gamma_0
        gamma_8 = Jx/gamma_0
        S = attrs['params']['S']
        b = attrs['params']['b']
        c = attrs['params']['c']
        rho = attrs['params']['rho']
        e = attrs['params']['e']
        S_prop = attrs['params']['S_prop']
        k_motor = attrs['params']['k_motor']
        #kT_p = attrs['params']['kT_p']
        #kOmega = attrs['params']['kOmega']
        Clong_coeffs = attrs['longitudinal_coeffs']
        Clateral_coeffs = attrs['lateral_coeffs']
        
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
            return (C1 - Cm0 - Cm_alpha * alpha - Cm_q * c * q * 0.5/Va)/Cm_delta_e
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
            CD_alpha_NL = CD_p + (CL0 + CL_alpha * alpha)**2/(np.pi * e * AR)
            CX  = -CD_alpha_NL * np.cos(alpha) + CL_alpha_NL * np.sin(alpha)
            CX_delta_e = -CD_delta_e * np.cos(alpha) + CL_delta_e * np.sin(alpha)
            CX_q = -CD_q * np.cos(alpha) + CL_q * np.sin(alpha)
            C2 = 2 * mass * (-r * v  + q * w + g * np.sin(theta))
            C3 = -2 * C0 * (CX + CX_q * c * q * 0.5/Va + CX_delta_e * delta_e )
            C4 = rho * C_prop * S_prop * k_motor**2
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
            
    def trim(self, Va, gamma, turn_radius, max_iters=5000, epsilon=1e-8, kappa=1e-6):
        R = turn_radius
        def J(alpha, beta, phi):        
            trimmed_state, trimmed_control = self.compute_trimmed_states_inputs(Va, gamma, turn_radius, alpha, beta, phi)
            f = self.f(trimmed_state, trimmed_control)
            f[0] = 0.
            f[1] = 0.
            
            xdot = np.zeros((12,), dtype = np.double)
            xdot[2] = -Va * np.sin(gamma)
            xdot[8] = Va/turn_radius * np.cos(gamma)
            J = np.linalg.norm(xdot[2:] - f[2:])**2
            return J
        def gradient_descent(alpha, beta, phi):
            iters = 0
            J0 = np.inf
            while(iters < max_iters):
                alpha_plus = alpha + epsilon
                beta_plus = beta + epsilon
                phi_plus = phi + epsilon
                J0 = J(alpha, beta, phi)
                dJ_dalpha = (J(alpha_plus, beta, phi) - J0)/epsilon
                dJ_dbeta = (J(alpha, beta_plus, phi) - J0)/epsilon
                dJ_dphi = (J(alpha, beta, phi_plus) - J0)/epsilon
                alpha = alpha - kappa * dJ_dalpha
                beta = beta - kappa * dJ_dbeta
                phi = phi - kappa * dJ_dphi
                iters += 1
                if iters%100==0:
                    print 'J: %f at iteration %d' % (J0, iters)
            return alpha, beta, phi
        alpha_0 = -0.0
        beta_0 = 0.
        phi_0 = 0.
        
        alpha, beta, phi = gradient_descent(alpha_0, beta_0, phi_0)
        trimmed_state, trimmed_control = self.compute_trimmed_states_inputs(Va, gamma, R, alpha, beta, phi)
        
        return trimmed_state, trimmed_control
    
    def f(self, y, control_input):
        mass = self.attrs['params']['mass']
        Jx = self.attrs['params']['Jx']
        Jy = self.attrs['params']['Jy']
        Jz = self.attrs['params']['Jz']
        Jxz = self.attrs['params']['Jxz']
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
        
        forces, moments = self.partial_forces_and_moments(y, control_input)
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
        dy[11] = gamma_7 * p * q - gamma_1 * q * r + gamma_4 * l + gamma_8 * n
        return dy
        
    def linearize(self, nominal_state, nominal_control_input, epsilon=1e-8):
        A = np.zeros((12, 12), dtype=np.double)
        B = np.zeros((12, 4), dtype=np.double)
        f_nominal =  self.f(nominal_state, nominal_control_input)        
        state_mask = np.zeros((12,), dtype=np.double)
        input_mask = np.zeros((4,), dtype=np.double)
        for col in range(12):
            state_mask[col] = 1.0
            f_ = self.f(nominal_state + epsilon * state_mask, nominal_control_input) 
            A[:, col] = (f_ - f_nominal)/epsilon
            state_mask[col] = 0.0
        for col in range(4):
            input_mask[col] = 1.0
            f_ = self.f(nominal_state, nominal_control_input + epsilon * input_mask) 
            B[:, col] = (f_ - f_nominal)/epsilon
            input_mask[col] = 0.0
        return A, B
        
    @property
    def control_inputs(self):
        return self._control_inputs
    @control_inputs.setter
    def control_inputs(self, inputs):
        self._control_inputs = inputs
        self.integrator.set_f_params(self.attrs, self._control_inputs, self.partial_forces_and_moments)
        
from abc import ABCMeta, abstractmethod
class  FixedWingUAVGuidanceModel(DynamicsBase):
    __metaclass__ = ABCMeta
    def __init__(self, x0, t0, dt_integration, attrs):
        self.attrs = attrs
        super(FixedWingUAVGuidanceModel, self).__init__(x0, t0, dt_integration)
        self.t0 = t0
        self.set_integrator(FixedWingUAVGuidanceModel.model, 'dop853', jac = None, rtol = 1e-8)
        
    def wind_model(self, Va):
        return 0.0, 0.0 , 0.0

    @abstractmethod
    def model(*args):
        pass

    def state(self):
        return self.model.x

    #factory method to generate the required model
    @staticmethod
    def generate_model(model_type, x0, t0, dt_integration, attrs):
        if model_type == 'course':
            return KinematicGuidanceModelWithCourse(x0, t0, dt_integration, attrs)
        elif model_type == 'roll':
            return KinematicGuidanceModelWithRoll(x0, t0, dt_integration, attrs)
        elif model_type == 'pitch':
            return KinematicGuidanceModelWithPitch(x0, t0, dt_integration, attrs)
        else:
            msg = 'model ' + model_type + 'not supported.\n'
            msg += 'supported model types are (course, roll, pitch)'
            raise Exception(msg)

class KinematicGuidanceModelWithCourse(FixedWingUAVGuidanceModel):
    def __init__(self, x0, t0, dt_integration, attrs):
        super(KinematicGuidanceModelWithCourse, self).__init__(x0, t0, dt_integration)

    @staticmethod
    def model(t, y, attrs, wind_model, Va_c, h_c, chi_c, h_dot_c, chi_dot_c):
        b_chi = attrs['b_chi']
        b_chi_dot = attrs['b_chi_dot']
        b_h = attrs['b_h']
        b_h_dot = attrs['b_h_dot']
        b_Va = attrs['b_Va']

        Va = y[6]
        chi = y[2]
        h = y[4]
        wn, we, wd = wind_model(Va)
        psi = chi - np.arcsin((-wn * np.sin(chi) + we * np.cos(chi))/Va)
        dy = np.zeros((7,), dtype = np.double)
        dy[0] = Va * np.cos(psi) + wn
        dy[1] = Va * np.sin(psi) + we
        dy[2] = y[3]
        dy[3] = b_chi_dot * (chi_dot_c - y[3]) + b_chi * (chi_c - chi)
        dy[4] = y[5]
        dy[5] = b_h_dot * (h_dot_c - y[5]) + b_h * (h_c - h)
        dy[6] = b_Va * (Va_c - Va)
        return dy

    def __call__(self, dt, Va_c, h_c, chi_c, h_dot_c = 0., chi_dot_c = 0.):
        self.integrator.set_f_params(self.attrs, self.wind_model, Va_c, h_c, chi_c, h_dot_c, chi_dot_c)
        self.integrate(dt + self.integrator.t)

class KinematicGuidanceModelWithRoll(FixedWingUAVGuidanceModel):
    def __init__(self, x0, t0, dt_integration, attrs):
        super(KinematicGuidanceModelWithCourse, self).__init__(x0, t0, dt_integration)

    @staticmethod
    def model(t, y, attrs, wind_model, Va_c, h_c, phi_c, h_dot_c):
        g = 9.81
        b_h = attrs['b_h']
        b_h_dot = attrs['b_h_dot']
        b_Va = attrs['b_Va']
        b_phi = attrs['b_phi']

        Va = y[5]
        phi = y[6]
        h = y[3]
        psi = y[2]
        wn, we, wd = wind_model(Va)
        dy = np.zeros((7,), dtype = np.double)
        dy[0] = Va * np.cos(psi) + wn
        dy[1] = Va * np.sin(psi) + we
        dy[2] = g * np.tan(phi)/Va
        dy[3] = y[4]
        dy[4] = b_h_dot * (h_dot_c - y[4]) + b_h * (h_c - h)
        dy[5] = b_Va * (Va_c - Va)
        dy[6] = b_phi * (phi_c - phi)
        return dy

    def __call__(self, dt, Va_c, h_c, phi_c, h_dot_c = 0.):
        self.integrator.set_f_params(self.attrs, self.wind_model, Va_c, h_c, phi_c, h_dot_c)
        self.integrate(dt + self.integrator.t)

class KinematicGuidanceModelWithPitch(FixedWingUAVGuidanceModel):
    def __init__(self, x0, t0, dt_integration, attrs):
        super(KinematicGuidanceModelWithCourse, self).__init__(x0, t0, dt_integration)

    @staticmethod
    def model(t, y, attrs, wind_model, Va_c, phi_c, pitch_c):
        g = 9.81
        b_pitch = attrs['b_pitch']
        b_Va = attrs['b_Va']
        b_phi = attrs['b_phi']

        Va = y[5]
        chi = y[3]
        phi = y[6]
        pitch = y[4]
        wn, we, wd = wind_model(Va)
        psi = chi - np.arcsin((-wn * np.sin(chi) + we * np.cos(chi))/Va)
        #compute Vg
        a = 1.
        b = -2. * (wn * np.cos(chi) * np.cos(pitch) + we * np.sin(chi) * np.cos(pitch) - wd * np.sin(pitch))
        Vw_sqrd = wn**2 + we**2 + wd**2
        c = Vw_sqrd - Va**2
        Vg = (-b + np.sqrt(b**2 - 4 * a * c))/(2. * a)
        #air mass referenced flight path angle
        num = Vg * np.sin(pitch) + wd
        den = Va
        gamma_a = np.arcsin(num/den)
        dy = np.zeros((7,), dtype = np.double)
        dy[0] = Va * np.cos(psi) + wn
        dy[1] = Va * np.sin(psi) + we
        dy[2] = Va * np.sin(gamma_a) - wd
        dy[3] = g * np.tan(phi) * np.cos(chi-psi)/Vg
        dy[4] = b_pitch * (pitch_c - pitch)
        dy[5] = b_Va * (Va_c - Va)
        dy[6] = b_phi * (phi_c - phi)
        return dy

    def __call__(self, dt, Va_c, phi_c, pitch_c):
        self.integrator.set_f_params(self.attrs, self.wind_model, Va_c, phi_c, pitch_c)
        self.integrate(dt + self.integrator.t)
            
            
            