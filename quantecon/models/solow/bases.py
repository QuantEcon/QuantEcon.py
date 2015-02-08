"""
Base class for representing a system of differential/difference equations
that needs to convert SymPy expressions into vectorized, NumPy-aware functions.

@author : davidrpugh

"""
from __future__ import division
import collections

import numpy as np
from scipy import linalg
import sympy as sym

# declare generic symbolic variables
t, X = sym.symbols('t'), sym.DeferredVector('X')


class SymbolicBase(object):
    """
    Base class for representing a system of differential/difference equations
    that needs to convert SymPy expressions into vectorized, NumPy-aware
    functions.

    """

    __numeric_jacobian = None

    __numeric_rhs = None

    __symbolic_jacobian = None

    _modules = [{'ImmutableMatrix': np.array}, 'numpy']

    @property
    def _numeric_jacobian(self):
        """Vectorized function for evaluating Jacobian matrix."""
        if self.__numeric_jacobian is None:
            expr = self.jacobian.subs(self._variable_subs)
            self.__numeric_jacobian = self._lambdify_factory(expr)
        return self.__numeric_jacobian

    @property
    def _numeric_rhs(self):
        """Vectorized function for evaluating right-hand side of system."""
        if self.__numeric_rhs is None:
            expr = self.rhs.subs(self._variable_subs)
            self.__numeric_rhs = self._lambdify_factory(expr)
        return self.__numeric_rhs

    @property
    def _symbolic_args(self):
        """List of symbolic arguments used to lambdify expressions."""
        return self._symbolic_vars + self._symbolic_params

    @property
    def _symbolic_params(self):
        """List of symbolic model parameters."""
        return sym.var(list(self.params.keys()))

    @property
    def _symbolic_vars(self):
        """List of symbolic model variables."""
        return [t, X]

    @property
    def _variable_subs(self):
        """Generic variable substitutions."""
        return dict(zip(self.dependent_vars, [X[i] for i in range(self.N)]))

    @property
    def dependent_vars(self):
        """
        Model dependent variables.

        :getter: Return the model dependent variables.
        :setter: Set new model dependent variables.
        :type: list

        """
        return self._dependent_variables

    @dependent_vars.setter
    def dependent_vars(self, symbols):
        """Set new list of dependent variables."""
        self._dependent_variables = symbols

    @property
    def independent_var(self):
        """
        Symbolic variable representing the independent variable.

        :getter: Return the symbol representing the independent variable.
        :setter: Set a new symbol to represent the independent variable.
        :type: sympy.Symbol

        """
        return self._independent_var

    @independent_var.setter
    def independent_var(self, symbol):
        """Set a new symbol to represent the independent variable."""
        self._independent_var = symbol

    @property
    def jacobian(self):
        """
        Symbolic Jacobian matrix of partial derivatives.

        :getter: Return the Jacobian matrix.
        :type: sympy.Matrix

        """
        if self.__symbolic_jacobian is None:
            self.__symbolic_jacobian = self.rhs.jacobian(self.dependent_vars)
        return self.__symbolic_jacobian

    @property
    def N(self):
        """
        Dimension of the symbolic system of equations.

        :getter: Return the current dimension.
        :type: int

        """
        return self.rhs.shape[0]

    @property
    def params(self):
        """
        Dictionary of model parameters.

        :getter: Return the current parameter dictionary.
        :setter: Set a new parameter dictionary.
        :type: dict

        """
        return self._params

    @params.setter
    def params(self, value):
        """Set a new parameter dictionary."""
        valid_params = self._validate_params(value)
        self._params = self._order_params(valid_params)

    @property
    def rhs(self):
        """
        Symbolic representation of the right-hand side of a system of
        differential/difference equations.

        :getter: Return the right-hand side of the system of equations.
        :setter: Set a new right-hand side of the system of equations.
        :type: sympy.Matrix

        """
        return self._rhs

    @rhs.setter
    def rhs(self, system):
        """Set a new right-hand side of the system of equations."""
        self._rhs = self._validate_rhs(system)
        self._clear_cache()

    def _clear_cache(self):
        """Clear cached values."""
        self.__numeric_jacobian = None
        self.__numeric_rhs = None
        self.__symbolic_jacobian = None

    def _lambdify_factory(self, expression):
        """Lambdify a symbolic expression."""
        return sym.lambdify(self._symbolic_args, expression, self._modules)

    @staticmethod
    def _order_params(params):
        """Cast a dictionary to an order dictionary."""
        return collections.OrderedDict(sorted(params.items()))

    @staticmethod
    def _validate_params(value):
        """Validate the dictionary of parameters."""
        if not isinstance(value, dict):
            mesg = "Attribute 'params' must have type dict, not {}"
            raise AttributeError(mesg.format(value.__class__))
        else:
            return value

    def _validate_rhs(self):
        raise NotImplementedError

    def compute_eigenvalues(self, t, X):
        """
        Compute the eigenvalues of the Jacobian matrix of partial derivatives
        evaluated at specific values of the independent variable, `t`, and
        dependent variables, `X`.

        Parameters
        ----------
        t : float
            Independent variable.
        X : numpy.ndarray (shape=(N,))
            Array of values for the `N` dependent variables.

        Returns
        -------
        eigvals : numpy.ndarray (shape=(N,))
            Array of eigenvalues of the Jacobian matrix.

        """
        eigvals = linalg.eigvals(self.evaluate_jacobian(t, X))
        return eigvals

    def evaluate_jacobian(self, t, X):
        """
        Return the Jacobian matrix of partial derivatives evaluated at specific
        values of the independent variable, `t`, and dependent variables, `X`.

        Parameters
        ----------
        t : float
            Independent variable.
        X : numpy.ndarray (shape=(N,))
            Array of values for the `N` dependent variables.

        Returns
        -------
        jac : numpy.ndarray (shape=(N,N))
            Evaluated Jacobian matrix.

        """
        jac = self._jacobian(t, X, *self.params.values())
        return jac

    def evaluate_rhs(self, t, X):
        """
        Return the right-hand side of a system of differential/difference
        equations evaluated at specific values of the independent variable,
        `t`, and dependent variables, `X`.

        Parameters
        ----------
        t : float
            Independent variable.
        X : numpy.ndarray (shape=(N,))
            Array of values for the `N` dependent variables.

        Returns
        -------
        rhs : numpy.ndarray (shape=(N,N))
            Evaluated Jacobian matrix.

        """
        rhs = self._rhs(t, X, *self.params.values())
        return rhs
