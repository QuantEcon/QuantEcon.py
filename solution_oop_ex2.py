class Polynomial:

    def __init__(self, coefficients):
        """
        Creates an instance of the Polynomial class representing 
        p(x) = a_0 x^0 + ... + a_N x^N, where a_i = coefficients[i].
        """
        self.coefficients = coefficients

    def evaluate(self, x):
        y = 0
        for i, a in enumerate(self.coefficients):
            y += a * x**i  
        return y

    def differentiate(self):
        new_coefficients = []
        for i, a in enumerate(self.coefficients):
            new_coefficients.append(i * a)
        # Remove the first element, which is zero
        del new_coefficients[0]  
        # And reset coefficients data to new values
        self.coefficients = new_coefficients

