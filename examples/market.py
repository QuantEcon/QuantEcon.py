"""
Filename: market.py
Reference: http://quant-econ.net/py/python_oop.html
"""

from __future__ import division
from scipy.integrate import quad

class Market:

    def __init__(self, ad, bd, az, bz, tax):
        """
        Set up market parameters.  All parameters are scalars.  See
        http://quant-econ.net/py/python_oop.html for interpretation.

        """
        self.ad, self.bd, self.az, self.bz, self.tax = ad, bd, az, bz, tax
        if ad < az:
            raise ValueError('Insufficient demand.')
        
    def price(self):
        "Return equilibrium price"
        return  (self.ad - self.az + self.bz*self.tax)/(self.bd + self.bz) 
    
    def quantity(self):
        "Compute equilibrium quantity"
        return  self.ad - self.bd * self.price()
        
    def area1(self):
        "Compute area under inverse demand function"
        a, error = quad(lambda x: (self.ad/self.bd) - (1/self.bd)* x, 0, self.quantity())
        return a
        
    def consumer_surp(self):
        "Compute consumer surplus"
        return  self.area1() - self.price() * self.quantity()
    
    def area2(self):
        "Compute area above the supply curve.  Note that we exclude the tax."
        a, error = quad(lambda x: -(self.az/self.bz) + (1/self.bz) * x, 0, self.quantity())  
        return a
    
    def producer_surp(self):
        "Compute producer surplus"
        return (self.price() - self.tax) * self.quantity() - self.area2()
    
    def taxrev(self):
        "Compute tax revenue"
        return self.tax * self.quantity()
        
    def inverse_demand(self,x):
        "Compute inverse demand"
        return self.ad/self.bd - (1/self.bd)* x
    
    def inverse_supply(self,x):
        "Compute inverse supply curve"
        return -(self.az/self.bz) + (1/self.bz) * x + self.tax
    
    def inverse_supply_no_tax(self,x):
        "Compute inverse supply curve without tax"
        return -(self.az/self.bz) + (1/self.bz) * x
    
    
    
