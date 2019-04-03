import numpy as np
from scipy.sparse.linalg import spsolve
from .auxFunctions import DT_X_D, DTinv_v, Dinv_v


class memoize(object):
    '''
    This class is a way to cache the result of a function, as well as its 1st and 2nd derivatives.
    SciPy's minimize() needs to be passed distinct objects for the jacobian + hessian, typically
    other functions that would be calculated separately. With Memoize, you can store this info
    from a function that returns [output, jac, hess] in this class object, then access the
    jacobian or hessian attribute with minimize().
    
    See the technique here: http://stackoverflow.com/a/17431749

    Args:
        a function that returns [value, jacobian, hessian]
    '''

    # Initialize class
    def __init__(self, fun):
        self.fun = fun
        self.value, self.jac, self.hess = None, None, None
        self.x = None

    # Calculate value of fun at x, along with jacobian and hessian
    def _compute(self, x, *args, **kwargs):
        self.x = np.asarray(x).copy()
        self.value, self.jac, self.hess = self.fun(x, *args, **kwargs)

    # Checks if x has changed since last call and, if so, recalculates
    # Returns value of function at x
    def __call__(self, x, *args, **kwargs):
        if self.value is not None and np.alltrue(x == self.x):
            return self.value
        else:
            self._compute(x, *args, **kwargs)
            return self.value

    # Checks if x has changed since last call and, if so, recalculates
    # Returns jacobian of function at x
    def jacobian(self, x, *args, **kwargs):
        if self.jac is not None and np.alltrue(x == self.x):
            return self.jac
        else:
            self._compute(x, *args, **kwargs)
            return self.jac

    # Checks if x has changed since last call and, if so, recalculates
    # Returns hessian of function at x (in the form of a dict)
    def hessian(self, x, *args, **kwargs):
        if self.hess is not None and np.alltrue(x == self.x):
            return self.hess
        else:
            self._compute(x, *args, **kwargs)
            return self.hess

    # Checks if x has changed since last call and, if so, recalculates
    # Returns product of hessian with arbitrary vector p (for optimization)
    def hessian_prod(self, x, p, *args, **kwargs):
        if self.hess is not None and np.alltrue(x == self.x):
            pass
        else:
            self._compute(x, *args, **kwargs)

        H = self.hess['H']
        K = self.hess['K']
        ddlogprior = self.hess['ddlogprior']

        center = DT_X_D(ddlogprior, K) + H

        return -DTinv_v(center @ Dinv_v(p, K), K)
