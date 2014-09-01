from __future__ import division

import numpy as np
from scipy import optimize
from scipy.integrate._ode import IntegratorBase

#from colloc_rk import crk

def forward_euler_step(self, f, y0, t0, t1, f_params):
    """Euler's method for approximating the solution of ODE."""
    # the basic Euler method with fixed step size
    y1 = y0 + (t1 - t0) * f(t0, y0, *f_params)  
        
    return [y1, t1]
    
class forward_euler(IntegratorBase):
    runner = forward_euler_step
    
    def __init__(self):
        self.success = True
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        y1, t = self.runner(f, y0, t0, t1, f_params)
        return y1, t

if forward_euler.runner:
    IntegratorBase.integrator_classes.append(forward_euler)
     
def backward_euler_step(self, f, jac, y0, t0, t1, f_params, jac_params):
    """Euler's implicit method for approximating the solution of ODE."""
    
    # the backward Euler method requires solving a non-linear equation in y1
    F = lambda y1: y1 - (y0 + (t1 - t0) * f(t1, y1, *f_params))  
    
    # create jacobian for F using jac
    F_jac = lambda y1: 1 - (t1 - t0) * jac(t1, y1, *jac_params)
    
    # use forward Euler to guess at the value y1
    guess_y1 = y0 + (t1 - t0) * f(t0, y0, *f_params) 
     
    # use root finding to solve the non-linear equation for y1
    res = optimize.root(F, guess_y1, method=self.method, jac=F_jac)
      
    # unpack the Result object
    y1      = res.x
    success = res.success
    mesg    = res.message
    
    return [y1, t1, success, mesg]

    
class backward_euler(IntegratorBase):
    runner = backward_euler_step
    
    def __init__(self, method='hybr'):
        self.method  = method
        self.success = True
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        args = [f, jac, y0, t0, t1, f_params, jac_params]
        y1, t, success, message = self.runner(*args)
        
        if success == False:
            self.success = False
            
        return y1, t

if backward_euler.runner:
    IntegratorBase.integrator_classes.append(backward_euler)
     
def trapezoidal_rule_step(self, f, jac, y0, t0, t1, f_params, jac_params):
    """Trapezoidal rule for approximating the solution of ODE."""
    
    # implicit method requires solving a non-linear equation in y1
    F = lambda y1: y1 - (y0 + (0.5 * (t1 - t0) * f(t0, y0, *f_params) + 
                               0.5 * (t1 - t0) * f(t1, y1, *f_params)))  
    
    # create jacobian for F using jac
    F_jac = lambda y1: 1 - 0.5 * (t1 - t0) * jac(t1, y1, *jac_params)
    
    # use forward Euler to guess at the value y1
    guess_y1 = y0 + (t1 - t0) * f(t0, y0, *f_params) 
     
    # use root finding to solve the non-linear equation for y1
    res = optimize.root(F, guess_y1, method=self.method, jac=F_jac)
      
    # unpack the Result object
    y1      = res.x
    success = res.success
    mesg    = res.message
    
    return [y1, t1, success, mesg]
    
class trapezoidal_rule(IntegratorBase):
    runner = trapezoidal_rule_step
    
    def __init__(self, method='hybr'):
        self.method  = method
        self.success = True
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        args = [f, jac, y0, t0, t1, f_params, jac_params]
        y1, t, success, message = self.runner(*args)
        
        if success == False:
            self.success = False
            
        return y1, t

if trapezoidal_rule.runner:
    IntegratorBase.integrator_classes.append(trapezoidal_rule)

############################# Runge-Kutta methods #############################    

########## Explicit Runge-Kutta methods ##########
def rk2_step(self, f, y0, t0, t1, f_params):
    """Second-order Runge-Kutta method."""
    # compute the step-size
    h = t1 - t0 
    
    # RK matrix
    A = np.array([[0, 0],
                  [1 / 2, 0]])
                 
    # RK nodes
    c = np.array([0, 1 / 2])
    
    # define the RK stages
    xi1 =  y0
    xi2 =  y0 + A[1,0] * h * f(t0 + c[0] * h, xi1, *f_params)
    
    # RK weights
    b = np.array([0, 1])
    
    # ERK2 rule is a weighted combination of the rk stages
    y1 = (y0 + h * b[0] * f(t0 + c[0] * h, xi1, *f_params) 
             + h * b[1] * f(t0 + c[1] * h, xi2, *f_params))
         
    return [y1, t1]
    
class rk2(IntegratorBase):
    runner = rk2_step
    
    def __init__(self):
        self.success = True
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        y1, t = self.runner(f, y0, t0, t1, f_params)
        return y1, t

if rk2.runner:
    IntegratorBase.integrator_classes.append(rk2)

def rk3_step(self, f, y0, t0, t1, f_params):
    """Classic third-order Runge-Kutta method."""
    # compute the step-size
    h = t1 - t0
    
    # RK matrix
    A = np.array([[0, 0, 0],
                  [1 / 2, 0, 0],
                  [-1, 2, 0]])
                 
    # RK nodes
    c = np.array([0, 1 / 2, 1])
    
    # define the RK stages
    xi1 =  y0
    xi2 =  y0 + A[1,0] * h * f(t0 + c[0] * h, xi1, *f_params)
    xi3 = (y0 + A[2,0] * h * f(t0 + c[0] * h, xi1, *f_params)
              + A[2,1] * h * f(t0 + c[1] * h, xi2, *f_params))
    
    
    # RK weights
    b = np.array([1 / 6, 2 / 3, 1 / 6])
    
    # ERK3 rule is a weighted combination of the rk stages
    y1 = (y0 + h * b[0] * f(t0 + c[0] * h, xi1, *f_params) 
             + h * b[1] * f(t0 + c[1] * h, xi2, *f_params) 
             + h * b[2] * f(t0 + c[2] * h, xi3, *f_params))
       
    return [y1, t1]
    
class rk3(IntegratorBase):
    runner = rk3_step
    
    def __init__(self):
        self.success = True
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        y1, t = self.runner(f, y0, t0, t1, f_params)
        return y1, t

if rk3.runner:
    IntegratorBase.integrator_classes.append(rk3)

def rk4_step(self, f, y0, t0, t1, f_params):
    """Classic fourth-order Runge-Kutta method."""
    # compute the step size
    h = t1 - t0
    
    # RK matrix
    A = np.array([[0, 0, 0, 0],
                  [1 / 2, 0, 0, 0],
                  [0, 1 / 2, 0, 0],
                  [0, 0, 1, 0]])
                 
    # RK nodes
    c = np.array([0, 1 / 2, 1 / 2, 1])
    
    # define the RK stages
    xi1 =  y0
    xi2 =  y0 + A[1,0] * h * f(t0 + c[0] * h, xi1, *f_params)
    xi3 = (y0 + A[2,0] * h * f(t0 + c[0] * h, xi1, *f_params)
              + A[2,1] * h * f(t0 + c[1] * h, xi2, *f_params))
    xi4 = (y0 + A[3,0] * h * f(t0 + c[0] * h, xi1, *f_params) 
              + A[3,1] * h * f(t0 + c[1] * h, xi2, *f_params) 
              + A[3,2] * h * f(t0 + c[2] * h, xi3, *f_params))
    
    # RK weights
    b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
    
    # RK4 rule is a weighted combination of the rk stages
    y1 = (y0 + h * b[0] * f(t0 + c[0] * h, xi1, *f_params) 
             + h * b[1] * f(t0 + c[1] * h, xi2, *f_params) 
             + h * b[2] * f(t0 + c[2] * h, xi3, *f_params)
             + h * b[3] * f(t0 + c[3] * h, xi4, *f_params))
    """
    h = (t1 - t0)
    
    z1 = f(t0, y0, *f_params)
    z2 = f(t0 + 0.5 * h, y0 + 0.5 * h * z1, *f_params)
    z3 = f(t0 + 0.5 * h, y0 + 0.5 * h * z2, *f_params)
    z4 = f(t0 + h, y0 + h * z3, *f_params)
    y1 = y0 + (h / 6) * (z1 + 2 * z2 + 2 * z3 + z4) 
    """
    return [y1, t1]
    
class rk4(IntegratorBase):
    runner = rk4_step
    
    def __init__(self):
        self.success = True
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        y1, t = self.runner(f, y0, t0, t1, f_params)
        return y1, t

if rk4.runner:
    IntegratorBase.integrator_classes.append(rk4)

def rk5_step(self, f, y0, t0, t1, f_params):
    """Fifth-order Runge-Kutta method."""    
    # compute the step-size
    h = t1 - t0
    
    # RK matrix
    A = np.array([[0, 0, 0, 0, 0, 0],
                  [1 / 3, 0, 0, 0, 0, 0],
                  [4 / 25, 6 / 25, 0, 0, 0, 0],
                  [1 / 4, -3, 15 / 4, 0, 0, 0],
                  [2 / 27, 10 / 9, -50 / 81, 8 / 81, 0, 0],
                  [2 / 25, 12 / 25, 2 / 15, 8 / 75, 0, 0]])
                 
    # RK nodes
    c = np.array([0, 1 / 3, 2 / 5, 1, 2 / 3, 4 / 5])
    
    # define the RK stages
    xi1 =  y0
    xi2 =  y0 + A[1,0] * h * f(t0 + c[0] * h, xi1, *f_params)
    xi3 = (y0 + A[2,0] * h * f(t0 + c[0] * h, xi1, *f_params)
              + A[2,1] * h * f(t0 + c[1] * h, xi2, *f_params))
    xi4 = (y0 + A[3,0] * h * f(t0 + c[0] * h, xi1, *f_params) 
              + A[3,1] * h * f(t0 + c[1] * h, xi2, *f_params) 
              + A[3,2] * h * f(t0 + c[2] * h, xi3, *f_params))
    xi5 = (y0 + A[4,0] * h * f(t0 + c[0] * h, xi1, *f_params) 
              + A[4,1] * h * f(t0 + c[1] * h, xi2, *f_params) 
              + A[4,2] * h * f(t0 + c[2] * h, xi3, *f_params)
              + A[4,3] * h * f(t0 + c[3] * h, xi4, *f_params))
    xi6 = (y0 + A[5,0] * h * f(t0 + c[0] * h, xi1, *f_params) 
              + A[5,1] * h * f(t0 + c[1] * h, xi2, *f_params) 
              + A[5,2] * h * f(t0 + c[2] * h, xi3, *f_params)
              + A[5,3] * h * f(t0 + c[3] * h, xi4, *f_params)
              + A[5,4] * h * f(t0 + c[4] * h, xi5, *f_params))
    
    # RK weights
    b = np.array([23 / 192, 0, 125 / 192, 0, -27 / 64, 125 / 192])
    
    # RK5 rule is a weighted combination of the rk stages
    y1 = (y0 + h * b[0] * f(t0 + c[0] * h, xi1, *f_params) 
             + h * b[1] * f(t0 + c[1] * h, xi2, *f_params) 
             + h * b[2] * f(t0 + c[2] * h, xi3, *f_params)
             + h * b[3] * f(t0 + c[3] * h, xi4, *f_params)
             + h * b[4] * f(t0 + c[4] * h, xi5, *f_params)
             + h * b[5] * f(t0 + c[5] * h, xi6, *f_params))
     
    return [y1, t1]
    
class rk5(IntegratorBase):
    runner = rk5_step
    
    def __init__(self):
        self.success = True
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        y1, t = self.runner(f, y0, t0, t1, f_params)
        return y1, t

if rk5.runner:
    IntegratorBase.integrator_classes.append(rk5)

def rk6_step(self, f, y0, t0, t1, f_params):
    """Sixth-order Runge-Kutta method from Butcher (2008)."""    
    # compute the step-size
    h = t1 - t0
    
    # RK matrix
    A = np.array([[0, 0, 0, 0, 0, 0, 0],
                  [1 / 3, 0, 0, 0, 0, 0, 0],
                  [0, 2 / 3, 0, 0, 0, 0, 0],
                  [1 / 12, 1 / 3, -1 / 12, 0, 0, 0, 0],
                  [25 / 48, -55 / 24, 35 / 48, 15 / 8, 0, 0, 0],
                  [3 / 20, -11 / 24, -1 / 8, 1 / 2, 1 / 10, 0, 0],
                  [-261 / 260, 33 / 13, 43 / 156, -118 / 39, 32 / 195, 80 / 39, 0]])
                 
    # RK nodes
    c = np.array([0, 1 / 3, 2 / 3, 1 / 3, 5 / 6, 1 / 6, 1])
    
    # define the RK stages
    xi1 =  y0
    xi2 =  y0 + A[1,0] * h * f(t0 + c[0] * h, xi1, *f_params)
    xi3 = (y0 + A[2,0] * h * f(t0 + c[0] * h, xi1, *f_params)
              + A[2,1] * h * f(t0 + c[1] * h, xi2, *f_params))
    xi4 = (y0 + A[3,0] * h * f(t0 + c[0] * h, xi1, *f_params) 
              + A[3,1] * h * f(t0 + c[1] * h, xi2, *f_params) 
              + A[3,2] * h * f(t0 + c[2] * h, xi3, *f_params))
    xi5 = (y0 + A[4,0] * h * f(t0 + c[0] * h, xi1, *f_params) 
              + A[4,1] * h * f(t0 + c[1] * h, xi2, *f_params) 
              + A[4,2] * h * f(t0 + c[2] * h, xi3, *f_params)
              + A[4,3] * h * f(t0 + c[3] * h, xi4, *f_params))
    xi6 = (y0 + A[5,0] * h * f(t0 + c[0] * h, xi1, *f_params) 
              + A[5,1] * h * f(t0 + c[1] * h, xi2, *f_params) 
              + A[5,2] * h * f(t0 + c[2] * h, xi3, *f_params)
              + A[5,3] * h * f(t0 + c[3] * h, xi4, *f_params)
              + A[5,4] * h * f(t0 + c[4] * h, xi5, *f_params))
    xi7 = (y0 + A[6,0] * h * f(t0 + c[0] * h, xi1, *f_params) 
              + A[6,1] * h * f(t0 + c[1] * h, xi2, *f_params) 
              + A[6,2] * h * f(t0 + c[2] * h, xi3, *f_params)
              + A[6,3] * h * f(t0 + c[3] * h, xi4, *f_params)
              + A[6,4] * h * f(t0 + c[4] * h, xi5, *f_params)
              + A[6,5] * h * f(t0 + c[5] * h, xi6, *f_params))
    
    # RK weights
    b = np.array([13 / 200, 0, 11 / 40, 11 / 40, 4 / 25, 4 / 25, 13 / 200])
    
    # RK5 rule is a weighted combination of the rk stages
    y1 = (y0 + h * b[0] * f(t0 + c[0] * h, xi1, *f_params) 
             + h * b[1] * f(t0 + c[1] * h, xi2, *f_params) 
             + h * b[2] * f(t0 + c[2] * h, xi3, *f_params)
             + h * b[3] * f(t0 + c[3] * h, xi4, *f_params)
             + h * b[4] * f(t0 + c[4] * h, xi5, *f_params)
             + h * b[5] * f(t0 + c[5] * h, xi6, *f_params)
             + h * b[6] * f(t0 + c[6] * h, xi7, *f_params))
     
    return [y1, t1]
    
class rk6(IntegratorBase):
    runner = rk6_step
    
    def __init__(self):
        self.success = True
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        y1, t = self.runner(f, y0, t0, t1, f_params)
        return y1, t

if rk6.runner:
    IntegratorBase.integrator_classes.append(rk6)

########## Implicit Runge-Kutta methods ##########
def irk_step(self, f, jac, y0, t0, t1, f_params, jac_params, A, b, c, method, **kwargs):
    """Implicit Runge-Kutta method."""
     
    # defines the system of nu * n non-linear algebraic equations
    def func(xi):
        out = (xi - (y0 + (t1 - t0) * A[0, 0] * 
                        f(t0 + c[0] * (t1 - t0), xi, *f_params)))
        return out
    
    # create jacobian for func using jac
    def func_jac(xi):
        out = A[0, 0] * (t1 - t0) * jac(t0 + c[0] * (t1 - t0), xi[0],  *jac_params)
        return out
    
    # solve system of n non-linear algebraic equations        
    res = optimize.root(func, guess_xi, method=method, jac=func_jac)
    
    # unpack the results object
    xi1     =res.x[0]
    success = res.success
    mesg    = res.message
    
    # irk1 rule is a weighted combination of the rk stages
    y1 = y0 + (t1 - t0) * b[0] * f(t0 + c[0], xi1, *f_params) 
         
    return [y1, t1, success, mesg]
    
class irk(IntegratorBase):
    runner = irk_step
    
    def __init__(self, nu, method='hybr', solver_options=None):
        
        # initialize attributes
        self.success        = True
        self.method         = method
        self.solver_options = None
         
        # RK nodes are the Gauss-Legendre roots on [0,1]
        coefs = np.zeros(nu + 1)
        coefs[-1] = 1.0
        P = np.polynomial.Legendre(coefs, domain=[0,1])
        self.c = P.roots()
    
        # RK matrix and weights 
        self.A, self.b = crk(self.c)
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        args = [f, jac, y0, t0, t1, f_params, jac_params, self.method]
        y1, t, success, message = self.runner(*args)
        
        if success == False:
            self.success = False
            
        return y1, t

if irk.runner:
    IntegratorBase.integrator_classes.append(rk2)

########## Embedded Runge-Kutta methods ##########
def erk12_step(self, f, y0, t0, t1, f_params, h, delta, beta):
    """Second-order embedded Runge-Kutta method."""
    # RK matrix
    A = np.array([[0, 0],
                  [1, 0]])
                 
    # RK nodes
    c = np.array([0, 1])
    
    # define the RK stages
    xi1 =  y0
    xi2 =  y0 + A[1,0] * (t1 - t0) * f(t0 + c[0] * (t1 - t0), xi1, *f_params)
    
    # RK weights for Heun's method (2nd order)
    b = np.array([1 / 2, 1 / 2])
    
    # Heun's method is a weighted combination of the rk stages
    y1 = (y0 + (t1 - t0) * b[0] * f(t0, xi1, *f_params) 
             + (t1 - t0) * b[1] * f(t0, xi2, *f_params))
             
    # RK weights for forward Euler method (1st order)
    b_tilde = np.array([1, 0])
    
    # forward Euler is a weighted combination of the rk stages
    y1_tilde = (y0 + (t1 - t0) * b_tilde[0] * f(t0, xi1, *f_params) 
                   + (t1 - t0) * b_tilde[1] * f(t0, xi2, *f_params))
         
    # estimate local truncation error
    e1 = y1 - y1_tilde
    
    # compute the next step size
    h1 = error_control_per_unit_step(e1, h, 2, delta, beta)
    
    return [y1, t1]
    
class erk12(IntegratorBase):
    runner        = erk12_step
    supports_step = True
    
    def __init__(self,
                 delta=1e-12,
                 rtol=1e-6, atol=1e-12,
                 nsteps=500,
                 max_step=0.0,
                 first_step=0.0,  
                 safety=0.9,
                 ifactor=10.0,
                 dfactor=0.2,
                 beta=0.9,
                 ):
        self.delta = delta
        self.rtol = rtol
        self.atol = atol
        self.nsteps = nsteps
        self.max_step = max_step
        self.first_step = first_step
        self.safety = safety
        self.ifactor = ifactor
        self.dfactor = dfactor
        self.beta = beta
        self.success = 1
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        y1, t = self.runner(f, y0, t0, t1, f_params)
        return y1, t
        
    def step(self, *args):
        itask = self.call_args[2]
        self.call_args[2] = 2
        r = self.run(*args)
        self.call_args[2] = itask
        return r

if erk12.runner:
    IntegratorBase.integrator_classes.append(erk12)

########################## Linear multi-step methods ##########################    

def ab2_step(self, f, y0, t0, t1, f_params):
    """2-step Adams-Bashforth method for approximating the solution of ODE."""
    # compute the step size
    h = (t1 - t0)
    
    # use rk2 to compute the starting values
    if len(self.past_values['y']) < 2:
        y1, t1 = rk2_step(self, f, y0, t0, t1, f_params)
        
        # remember theses values
        self.past_values['y'].append(y1)
        self.past_values['t'].append(t1)
        
        return [y1, t1]
        
    # 2nd order Adams-Bashford
    else:
        # recall previous values
        y0, y1 = self.past_values['y']
        t0, t1 = self.past_values['t']
        
        y2 = y1 + h * ((3 / 2) * f(t1, y1, *f_params) - 
                       (1 / 2) * f(t0, y0, *f_params))               
        t2 = t1 + h
        
        # erase some memory
        self.past_values['y'].pop(0)
        self.past_values['t'].pop(0)
        
        # remember current values
        self.past_values['y'].append(y2) 
        self.past_values['t'].append(t2)
          
        return [y2, t2]

class ab2(IntegratorBase):
    runner = ab2_step
    
    def __init__(self):
        self.success     = True
        self.past_values = {'y':[], 't':[]}
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        y1, t = self.runner(f, y0, t0, t1, f_params)
        return y1, t

if ab2.runner:
    IntegratorBase.integrator_classes.append(ab2)

def ab3_step(self, f, y0, t0, t1, f_params):
    """3-step Adams-Bashforth method for approximating the solution of ODE."""
    # compute the step size
    h = (t1 - t0)
    
    # use rk3 to compute the starting values
    if len(self.past_values['y']) < 3:
        y1, t1 = rk3_step(self, f, y0, t0, t1, f_params)
        
        # remember theses values
        self.past_values['y'].append(y1)
        self.past_values['t'].append(t1)
        
        return [y1, t1]
        
    # 3rd order Adams-Bashford
    else:
        # recall previous values
        y0, y1, y2 = self.past_values['y']
        t0, t1, t2 = self.past_values['t']
        
        y3 = y2 + h * ((23 / 12) * f(t2, y2, *f_params) - 
                       (4 / 3) * f(t1, y1, *f_params) + 
                       (5 / 12) * f(t0, y0, *f_params))               
        t3 = t2 + h
        
        # erase memory
        self.past_values['y'].pop(0)
        self.past_values['t'].pop(0)
        
        # remember current values
        self.past_values['y'].append(y3) 
        self.past_values['t'].append(t3)
          
        return [y3, t3]

class ab3(IntegratorBase):
    runner = ab3_step
    
    def __init__(self):
        self.success     = True
        self.past_values = {'y':[], 't':[]}
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        y1, t = self.runner(f, y0, t0, t1, f_params)
        return y1, t

if ab3.runner:
    IntegratorBase.integrator_classes.append(ab3)

def ab4_step(self, f, y0, t0, t1, f_params):
    """4-step Adams-Bashforth method for approximating the solution of ODE."""
    # compute the step size
    h = (t1 - t0)
    
    # use rk4 to compute the starting values
    if len(self.past_values['y']) < 4:
        y1, t1 = rk4_step(self, f, y0, t0, t1, f_params)
        
        # remember theses values
        self.past_values['y'].append(y1)
        self.past_values['t'].append(t1)
        
        return [y1, t1]
    
    # 4th order Adams-Bashford
    else:
        # recall previous values
        y0, y1, y2, y3 = self.past_values['y']
        t0, t1, t2, t3 = self.past_values['t']
        
        y4 = y3 + h * ((55 / 24) * f(t3, y3, *f_params) - 
                       (59 / 24) * f(t2, y2, *f_params) + 
                       (37 / 24) * f(t1, y1, *f_params) - 
                       (3 / 8) * f(t0, y0, *f_params))               
        t4 = t3 + h
        
        # erase memory
        self.past_values['y'].pop(0)
        self.past_values['t'].pop(0)
        
        # remember current values
        self.past_values['y'].append(y4) 
        self.past_values['t'].append(t4)
          
        return [y4, t4]

class ab4(IntegratorBase):
    runner = ab4_step
    
    def __init__(self):
        self.success     = True
        self.past_values = {'y':[], 't':[]}
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        y1, t = self.runner(f, y0, t0, t1, f_params)
        return y1, t

if ab4.runner:
    IntegratorBase.integrator_classes.append(ab4)

def ab5_step(self, f, y0, t0, t1, f_params):
    """5-step Adams-Bashforth method for approximating the solution of ODE."""
    # compute the step size
    h = (t1 - t0)
    
    # use rk5 to compute the starting values
    if len(self.past_values['y']) < 5:
        y1, t1 = rk5_step(self, f, y0, t0, t1, f_params)
        
        # remember theses values
        self.past_values['y'].append(y1)
        self.past_values['t'].append(t1)
        
        return [y1, t1]
    
    # 5th order Adams-Bashford
    else:
        # recall previous values
        y0, y1, y2, y3, y4 = self.past_values['y']
        t0, t1, t2, t3, t4 = self.past_values['t']
        
        y5 = y4 + h * ((1901 / 720) * f(t4, y4, *f_params) - 
                       (1387 / 360) * f(t3, y3, *f_params) + 
                       (109 / 30) * f(t2, y2, *f_params) - 
                       (637 / 360) * f(t1, y1, *f_params) + 
                       (251 / 720) * f(t0, y0, *f_params))               
        t5 = t4 + h
        
        # erase memory
        self.past_values['y'].pop(0)
        self.past_values['t'].pop(0)
        
        # remember current values
        self.past_values['y'].append(y5) 
        self.past_values['t'].append(t5)
          
        return [y5, t5]

class ab5(IntegratorBase):
    runner = ab5_step
    
    def __init__(self):
        self.success     = True
        self.past_values = {'y':[], 't':[]}
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        y1, t = self.runner(f, y0, t0, t1, f_params)
        return y1, t

if ab5.runner:
    IntegratorBase.integrator_classes.append(ab5)

def am2_step(self, f, jac, y0, t0, t1, f_params, jac_params):
    """2-step Adams-Moulton method for approximating the solution of ODE."""
    # compute the step size
    h = (t1 - t0)
    
    # use rk3 to compute the starting values
    if len(self.past_values['y']) < 2:
        y1, t1 = rk3_step(self, f, y0, t0, t1, f_params)
        
        # remember theses values
        self.past_values['y'].append(y1)
        self.past_values['t'].append(t1)
        
        return [y1, t1, True, None]
        
    else:
        # recall previous values
        y0, y1 = self.past_values['y']
        t0, t1 = self.past_values['t']
        
        # compute next time step
        t2 = t1 + h
        
        # implicit method requires solving a non-linear equation in y1
        F = lambda y2: y2 - (y1 + h * ((5 / 12) * f(t2, y2, *f_params) + 
                                       (2 / 3) * f(t1, y1, *f_params) - 
                                       (1 / 12) * f(t0, y0, *f_params)))  
    
        # create jacobian for F using jac
        F_jac = lambda y2: 1 - h * (5 / 12) * jac(t2, y2, *jac_params)
    
        # use 2nd order Adams-Bashford to predict y2
        guess_y2 = y1 + h * ((3 / 2) * f(t1, y1, *f_params) - 
                             (1 / 2) * f(t1, y1, *f_params))
                            
        # use root finding to solve the non-linear equation for y2
        res = optimize.root(F, guess_y2, method=self.method, jac=F_jac)
      
        # unpack the Result object
        y2      = res.x
        success = res.success
        mesg    = res.message
        
        # erase memory
        self.past_values['y'].pop(0)
        self.past_values['t'].pop(0)
        
        # remember current values
        self.past_values['y'].append(y2) 
        self.past_values['t'].append(t2)
          
        return [y2, t2, success, mesg]

class am2(IntegratorBase):
    runner = am2_step
    
    def __init__(self, method='hybr'):
        self.success     = True
        self.method      = method
        self.past_values = {'y':[], 't':[]}
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        args = [f, jac, y0, t0, t1, f_params, jac_params]
        y1, t, success, message = self.runner(*args)
        
        if success == False:
            self.success = False
            
        return y1, t
        
if am2.runner:
    IntegratorBase.integrator_classes.append(am2)

def am3_step(self, f, jac, y0, t0, t1, f_params, jac_params):
    """3-step Adams-Moulton method for approximating the solution of ODE."""
    # compute the step size
    h = (t1 - t0)
    
    # use rk4 to compute the starting values
    if len(self.past_values['y']) < 3:
        y1, t1 = rk4_step(self, f, y0, t0, t1, f_params)
        
        # remember theses values
        self.past_values['y'].append(y1)
        self.past_values['t'].append(t1)
        
        return [y1, t1, True, None]
        
    else:
        # recall previous values
        y0, y1, y2 = self.past_values['y']
        t0, t1, t2 = self.past_values['t']
        
        # compute next time step
        t3 = t2 + h
        
        # implicit method requires solving a non-linear equation in y1
        F = lambda y3: y3 - (y2 + h * ((3 / 8) * f(t3, y3, *f_params) + 
                                       (19 / 24) * f(t2, y2, *f_params) - 
                                       (5 / 24) * f(t1, y1, *f_params) + 
                                       (1 / 24) * f(t0, y0, *f_params)))  
    
        # create jacobian for F using jac
        F_jac = lambda y3: 1 - h * (3 / 8) * jac(t3, y3, *jac_params)
    
        # use 3rd order Adams-Bashford to predict y3
        guess_y3 = y2 + h * ((23 / 12) * f(t2, y2, *f_params) - 
                             (4 / 3) * f(t1, y1, *f_params) + 
                             (5 / 12) * f(t0, y0, *f_params))   
                            
        # use root finding to solve the non-linear equation for y3
        res = optimize.root(F, guess_y3, method=self.method, jac=F_jac)
      
        # unpack the Result object
        y3      = res.x
        success = res.success
        mesg    = res.message
        
        # erase memory
        self.past_values['y'].pop(0)
        self.past_values['t'].pop(0)
        
        # remember current values
        self.past_values['y'].append(y3) 
        self.past_values['t'].append(t3)
          
        return [y3, t3, success, mesg]

class am3(IntegratorBase):
    runner = am3_step
    
    def __init__(self, method='hybr'):
        self.success     = True
        self.method      = method
        self.past_values = {'y':[], 't':[]}
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        args = [f, jac, y0, t0, t1, f_params, jac_params]
        y1, t, success, message = self.runner(*args)
        
        if success == False:
            self.success = False
            
        return y1, t
        
if am3.runner:
    IntegratorBase.integrator_classes.append(am3)

def am4_step(self, f, jac, y0, t0, t1, f_params, jac_params):
    """4-step Adams-Moulton method for approximating the solution of ODE."""
    # compute the step size
    h = (t1 - t0)
    
    # use rk5 to compute the starting values
    if len(self.past_values['y']) < 4:
        y1, t1 = rk5_step(self, f, y0, t0, t1, f_params)
        
        # remember theses values
        self.past_values['y'].append(y1)
        self.past_values['t'].append(t1)
        
        return [y1, t1, True, None]
        
    else:
        # recall previous values
        y0, y1, y2, y3 = self.past_values['y']
        t0, t1, t2, t3 = self.past_values['t']
        
        # compute next time step
        t4 = t3 + h
        
        # implicit method requires solving a non-linear equation in y4
        F = lambda y4: y4 - (y3 + h * ((251 / 720) * f(t4, y4, *f_params) + 
                                       (646 / 720) * f(t3, y3, *f_params) - 
                                       (264 / 720) * f(t2, y2, *f_params) + 
                                       (106 / 720) * f(t1, y1, *f_params) -
                                       (19 / 720) * f(t0, y0, *f_params)))  
    
        # create jacobian for F using jac
        F_jac = lambda y4: 1 - h * (251 / 720) * jac(t4, y4, *jac_params)
    
        # use 4th order Adams-Bashford to predict y4
        guess_y4 = y3 + h * ((55 / 24) * f(t3, y3, *f_params) - 
                             (59 / 24) * f(t2, y2, *f_params) + 
                             (37 / 24) * f(t1, y1, *f_params) - 
                             (3 / 8) * f(t0, y0, *f_params))   
                            
        # use root finding to solve the non-linear equation for y1
        res = optimize.root(F, guess_y4, method=self.method, jac=F_jac)
      
        # unpack the Result object
        y4      = res.x
        success = res.success
        mesg    = res.message
        
        # erase memory
        self.past_values['y'].pop(0)
        self.past_values['t'].pop(0)
        
        # remember current values
        self.past_values['y'].append(y4) 
        self.past_values['t'].append(t4)
          
        return [y4, t4, success, mesg]

class am4(IntegratorBase):
    runner = am4_step
    
    def __init__(self, method='hybr'):
        self.success     = True
        self.method      = method
        self.past_values = {'y':[], 't':[]}
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        args = [f, jac, y0, t0, t1, f_params, jac_params]
        y1, t, success, message = self.runner(*args)
        
        if success == False:
            self.success = False
            
        return y1, t
        
if am4.runner:
    IntegratorBase.integrator_classes.append(am4)
    
def bdf2_step(self, f, jac, y0, t0, t1, f_params, jac_params):
    """2-step BDF method for approximating the solution of ODE."""
    # compute the step size
    h = (t1 - t0)
    
    # use rk3 to compute the starting values
    if len(self.past_values['y']) < 2:
        y1, t1 = rk3_step(self, f, y0, t0, t1, f_params)
        
        # remember theses values
        self.past_values['y'].append(y1)
        self.past_values['t'].append(t1)
        
        return [y1, t1, True, None]
        
    else:
        # recall previous values
        y0, y1 = self.past_values['y']
        t0, t1 = self.past_values['t']
        
        # compute next time step
        t2 = t1 + h
        
        # implicit method requires solving a non-linear equation in y1
        F = lambda y2: (y2 - (4 / 3) * y1 + (1 / 3) * y0 - 
                             (2 / 3) * h * f(t2, y2, *f_params))
    
        # create jacobian for F using jac
        F_jac = lambda y2: 1 - (2 / 3) * h * jac(t2, y2, *jac_params)
    
        # use 2nd order Adams-Bashford to predict y2
        guess_y2 = y1 + h * ((3 / 2) * f(t1, y1, *f_params) - 
                             (1 / 2) * f(t1, y1, *f_params))
                            
        # use root finding to solve the non-linear equation for y2
        res = optimize.root(F, guess_y2, method=self.method, jac=F_jac)
      
        # unpack the Result object
        y2      = res.x
        success = res.success
        mesg    = res.message
        
        # erase memory
        self.past_values['y'].pop(0)
        self.past_values['t'].pop(0)
        
        # remember current values
        self.past_values['y'].append(y2) 
        self.past_values['t'].append(t2)
          
        return [y2, t2, success, mesg]

class bdf2(IntegratorBase):
    runner = bdf2_step
    
    def __init__(self, method='hybr'):
        self.success     = True
        self.method      = method
        self.past_values = {'y':[], 't':[]}
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        args = [f, jac, y0, t0, t1, f_params, jac_params]
        y1, t, success, message = self.runner(*args)
        
        if success == False:
            self.success = False
            
        return y1, t
        
if bdf2.runner:
    IntegratorBase.integrator_classes.append(bdf2)

def bdf3_step(self, f, jac, y0, t0, t1, f_params, jac_params):
    """3-step BDF method for approximating the solution of ODE."""
    # compute the step size
    h = (t1 - t0)
    
    # use rk4 to compute the starting values
    if len(self.past_values['y']) < 3:
        y1, t1 = rk4_step(self, f, y0, t0, t1, f_params)
        
        # remember theses values
        self.past_values['y'].append(y1)
        self.past_values['t'].append(t1)
        
        return [y1, t1, True, None]
        
    else:
        # recall previous values
        y0, y1, y2 = self.past_values['y']
        t0, t1, t2 = self.past_values['t']
        
        # compute next time step
        t3 = t2 + h
        
        # implicit method requires solving a non-linear equation in y1
        F = lambda y3: (y3 - (18 / 11) * y2 + (9 / 11) * y1 - (2 / 11) * y0 - 
                             (6 / 11) * h * f(t3, y3, *f_params))
    
        # create jacobian for F using jac
        F_jac = lambda y3: 1 - (6 / 11) * h * jac(t3, y3, *jac_params)
    
        # use 3rd order Adams-Bashford to predict y3
        guess_y3 = y2 + h * ((23 / 12) * f(t2, y2, *f_params) - 
                             (4 / 3) * f(t1, y1, *f_params) + 
                             (5 / 12) * f(t0, y0, *f_params))
                            
        # use root finding to solve the non-linear equation for y2
        res = optimize.root(F, guess_y3, method=self.method, jac=F_jac)
      
        # unpack the Result object
        y3      = res.x
        success = res.success
        mesg    = res.message
        
        # erase memory
        self.past_values['y'].pop(0)
        self.past_values['t'].pop(0)
        
        # remember current values
        self.past_values['y'].append(y3) 
        self.past_values['t'].append(t3)
          
        return [y3, t3, success, mesg]

class bdf3(IntegratorBase):
    runner = bdf3_step
    
    def __init__(self, method='hybr'):
        self.success     = True
        self.method      = method
        self.past_values = {'y':[], 't':[]}
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        args = [f, jac, y0, t0, t1, f_params, jac_params]
        y1, t, success, message = self.runner(*args)
        
        if success == False:
            self.success = False
            
        return y1, t
        
if bdf3.runner:
    IntegratorBase.integrator_classes.append(bdf3)

def bdf4_step(self, f, jac, y0, t0, t1, f_params, jac_params):
    """4-step BDF method for approximating the solution of ODE."""
    # compute the step size
    h = (t1 - t0)
    
    # use rk5 to compute the starting values
    if len(self.past_values['y']) < 4:
        y1, t1 = rk5_step(self, f, y0, t0, t1, f_params)
        
        # remember theses values
        self.past_values['y'].append(y1)
        self.past_values['t'].append(t1)
        
        return [y1, t1, True, None]
        
    else:
        # recall previous values
        y0, y1, y2, y3 = self.past_values['y']
        t0, t1, t2, t3 = self.past_values['t']
        
        # compute next time step
        t4 = t3 + h
        
        # implicit method requires solving a non-linear equation in y4
        F = lambda y4: (y4 - (48 / 25) * y3 + (36 / 25) * y2 - (16 / 25) * y1 + 
                             (3 / 25) * y0  - (12 / 25) * h * f(t4, y4, *f_params))
    
        # create jacobian for F using jac
        F_jac = lambda y4: 1 - (12 / 25) * h * jac(t4, y4, *jac_params)
    
        # use 4th order Adams-Bashford to predict y4
        guess_y4 = y3 + h * ((55 / 24) * f(t3, y3, *f_params) - 
                             (59 / 24) * f(t2, y2, *f_params) + 
                             (37 / 24) * f(t1, y1, *f_params) - 
                             (3 / 8) * f(t0, y0, *f_params)) 
                            
        # use root finding to solve the non-linear equation for y4
        res = optimize.root(F, guess_y4, method=self.method, jac=F_jac)
      
        # unpack the Result object
        y4      = res.x
        success = res.success
        mesg    = res.message
        
        # erase memory
        self.past_values['y'].pop(0)
        self.past_values['t'].pop(0)
        
        # remember current values
        self.past_values['y'].append(y4) 
        self.past_values['t'].append(t4)
          
        return [y4, t4, success, mesg]

class bdf4(IntegratorBase):
    runner = bdf4_step
    
    def __init__(self, method='hybr'):
        self.success     = True
        self.method      = method
        self.past_values = {'y':[], 't':[]}
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        args = [f, jac, y0, t0, t1, f_params, jac_params]
        y1, t, success, message = self.runner(*args)
        
        if success == False:
            self.success = False
            
        return y1, t
        
if bdf4.runner:
    IntegratorBase.integrator_classes.append(bdf4)

def bdf5_step(self, f, jac, y0, t0, t1, f_params, jac_params):
    """5-step BDF method for approximating the solution of ODE."""
    # compute the step size
    h = (t1 - t0)
    
    # use rk6 to compute the starting values
    if len(self.past_values['y']) < 5:
        y1, t1 = rk6_step(self, f, y0, t0, t1, f_params)
        
        # remember theses values
        self.past_values['y'].append(y1)
        self.past_values['t'].append(t1)
        
        return [y1, t1, True, None]
        
    else:
        # recall previous values
        y0, y1, y2, y3, y4 = self.past_values['y']
        t0, t1, t2, t3, t4 = self.past_values['t']
        
        # compute next time step
        t5 = t4 + h
        
        # implicit method requires solving a non-linear equation in y5
        F = lambda y5: (y5 - (300 / 137) * y4 + (300 / 137) * y3 - 
                             (200 / 137) * y2 + (75 / 137) * y1  - 
                             (12 / 137) * y0 - (60 / 137) * h * f(t5, y5, *f_params))
    
        # create jacobian for F using jac
        F_jac = lambda y5: 1 - (60 / 137) * h * jac(t5, y5, *jac_params)
    
        # use 5th order Adams-Bashford to predict y5
        guess_y5 = y4 + h * ((1901 / 720) * f(t4, y4, *f_params) - 
                             (1387 / 360) * f(t3, y3, *f_params) + 
                             (109 / 30) * f(t2, y2, *f_params) - 
                             (637 / 360) * f(t1, y1, *f_params) + 
                             (251 / 720) * f(t0, y0, *f_params)) 
                            
        # use root finding to solve the non-linear equation for y5
        res = optimize.root(F, guess_y5, method=self.method, jac=F_jac)
      
        # unpack the Result object
        y5      = res.x
        success = res.success
        mesg    = res.message
        
        # erase memory
        self.past_values['y'].pop(0)
        self.past_values['t'].pop(0)
        
        # remember current values
        self.past_values['y'].append(y5) 
        self.past_values['t'].append(t5)
          
        return [y5, t5, success, mesg]

class bdf5(IntegratorBase):
    runner = bdf5_step
    
    def __init__(self, method='hybr'):
        self.success     = True
        self.method      = method
        self.past_values = {'y':[], 't':[]}
        
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        args = [f, jac, y0, t0, t1, f_params, jac_params]
        y1, t, success, message = self.runner(*args)
        
        if success == False:
            self.success = False
            
        return y1, t
        
if bdf5.runner:
    IntegratorBase.integrator_classes.append(bdf5)
    
########################### Error-control functions ###########################

def error_control_per_unit_step(trunc_error, h, y0, p, rtol, atol, beta=0.9):
    """
    Adaptive step size control using error control per unit step. See 
    Dormand and Prince (1980) for details.
    
    Arguments:
        
        trunc_error: (array-like) Array of values representing a component-wise
                     estimate of the local truncation error.
                     
        h:           (float) Current step size.
        
        y0:          (array) Current value of y(t).
        
        p:           (int) Highest order of the two methods being used in the
                     adaptive step size scheme.
                     
        rtol:        (float) User-specified relative error tolerance.
        
        atol:        (float) User-specified absolute error tolerance.
        
        beta:        (float) Control parameter. Must be 0 < beta < 1. Default is
                     beta = 0.9.
                     
    Returns:
        
        new_h: (float) New step size.
                     
    """
    # delta may be an array!
    delta = rtol * np.abs(y0) + atol
    
    # update h according to the formula from Dormand and Prince (1980)
    new_h = beta * h * (delta / np.max(np.abs(trunc_error / h)))**(1 / p)
    
    return new_h
    