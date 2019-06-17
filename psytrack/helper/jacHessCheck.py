import numpy as np
from scipy.sparse.linalg import spsolve, inv


def jacHessCheck(fun, x0, *args, **kwargs):
    """
    Checks the accuracy of the analytic Jacobian and Hessian
    of a function that is part of the Memoize class by calculating
    finite differences in a tiny random direction.

    Args:
        fun : a function returning [result, jac, hess] that has
            been passed through a Memoize class
        x0 : the point from which to evaluate the function

    Returns:
        Nothing, simply prints out the analytic and finite
        differencing results for both the jac and hess
    """

    fun(x0, *args, **kwargs)
    JJ = fun.jacobian(x0, *args, **kwargs)
    HH = fun.hessian(x0, *args, **kwargs)

    tol = 1e-8
    randjump = np.random.random(size=x0.shape) * tol

    f1 = fun(x0 - randjump / 2, *args, **kwargs)
    JJ1 = fun.jacobian(x0 - randjump / 2, *args, **kwargs)

    f2 = fun(x0 + randjump / 2, *args, **kwargs)
    JJ2 = fun.jacobian(x0 + randjump / 2, *args, **kwargs)

    print("Analytic Jac:", np.dot(randjump, JJ))
    print("Finite Jac:  ", f2 - f1)

    if type(HH) is dict:
        print("Analytic Hess:",
              np.sum(fun.hessian_prod(x0, randjump, *args, **kwargs)))
    else:
        print("Analytic Hess:", np.sum(HH @ randjump))
    print("Finite Hess:  ", np.sum(JJ2 - JJ1))


def jacEltsCheck(fun, ind, x0, *args, **kwargs):
    """
    Checks the accuracy of individual elements in the analytic Jacobian
    of a function that is part of the Memoize class by calculating
    finite differences in the specified direction.

    Args:
        fun : a function returning [result, jac, hess] that has
            been passed through a Memoize class
        x0 : the point from which to evaluate the function

    Returns:
        Nothing, simply prints out the analytic and finite
        differencing results for the jacobian
    """

    fun(x0, *args, **kwargs)
    JJ = fun.jacobian(x0, *args, **kwargs)

    eps = 1e-5
    mask = np.zeros(len(x0))
    mask[ind] = eps

    f1 = fun(x0 - mask, *args, **kwargs)
    f2 = fun(x0 + mask, *args, **kwargs)

    dJ = (f2 - f1) / 2 / eps

    if np.sqrt((JJ[ind] - dJ)**2) > 1e-8:
        print(ind, ": ", np.sqrt((JJ[ind] - dJ)**2))
        print("Analytic Jac:", JJ[ind])
        print("Finite Jac:  ", dJ)


def hessEltsCheck(fun, ind, x0, *args, **kwargs):
    """
    Checks the accuracy of individual elements in the analytic Hessian
    of a function that is part of the Memoize class by calculating
    finite differences in the specified direction.

    Args:
        fun : a function returning [result, jac, hess] that has
            been passed through a Memoize class
        x0 : the point from which to evaluate the function

    Returns:
        Nothing, simply prints out the analytic and finite
        differencing results for the hessian
    """

    fun(x0, *args, **kwargs)
    HH = fun.hessian(x0, *args, **kwargs)

    if type(HH) is dict:
        if HH["P"].shape[0] > 1000:
            print("Hessian is too large (>1000) to check elements")
            return
        else:
            P = HH["P"]
            H = HH["H"]
            ddlogprior = HH["ddlogprior"]
            HH = -inv(P.T) @ H @ inv(P) - ddlogprior

    eps = 1e-4
    mask = np.zeros((len(x0), 2))
    mask[ind[0], 0] = eps
    mask[ind[1], 1] = eps
    v11 = fun(x0 + (mask @ np.array([[-1], [-1]]))[:, 0], *args, **kwargs)
    v21 = fun(x0 + (mask @ np.array([[1], [-1]]))[:, 0], *args, **kwargs)
    v12 = fun(x0 + (mask @ np.array([[-1], [1]]))[:, 0], *args, **kwargs)
    v22 = fun(x0 + (mask @ np.array([[1], [1]]))[:, 0], *args, **kwargs)

    dH = ((v22 - v21) - (v12 - v11)) / 4 / eps**2

    if (HH[ind[0], ind[1]] - dH)**2 > 1e-8:
        print(ind[0], ind[1], " : ", HH[ind[0], ind[1]] - dH)
        print("Analytic Hess:", HH[ind[0], ind[1]])
        print("Finite Hess:  ", dH)
