import numpy as np
from scipy.optimize import minimize

from psytrack.aux.memoize import memoize
from psytrack.aux.jacHessCheck import jacHessCheck
from psytrack.aux.auxFunctions import (
    DT_X_D,
    sparse_logdet,
    read_input,
    make_invSigma,
    DTinv_v,
    myblk_diags,
)


def getMAP(dat, hyper, weights, method=None, E0=None, showOpt=0):
    """Estimates epsilon parameters with random walk prior

    Args:
        dat : dict, all data from a specific subject
        hyper : a dictionary of hyperparameters used to construct the prior
            Must at least include sigma, can also include sigInit, sigDay
        weights : dict, name and count of weights in dat['inputs'] to fit
        method : str, control over type of learning, defaults to standard 
            trial-by-trial fitting; '_days' and '_constant' also supported
        E0 : initial parameter estimate, must be of approprite size N*K, 
            defaults to zeros
        showOpt : {0 : no text, 1 : verbose, 
            2+ : Hess + deriv check, done showOpt-1 times}
        maxIter : max iterations of the optimizer

    Returns:
        wMode : MAP estimate of the weights
        Hess : the Hessian of the log posterior at wMode, used for Laplace appx.
            in evidence max in this case, is a dict of sparse terms needed to
            construct Hess (which is not sparse)
        logEvd : log of the evidence
        llstruct : dictionary containing the components of the log evidence and
            other info
    """

    # -----
    # Initializations and Sanity Checks
    # -----

    # Check and count trials
    if "inputs" not in dat or "y" not in dat or type(
            dat["inputs"]) is not dict:
        raise Exception("getMAP_PBups: insufficient input, missing y")
    N = len(dat["y"])

    # Check and count weights
    K = 0
    if type(weights) is not dict:
        raise Exception("weights must be a dict")
    for i in weights.keys():
        if type(weights[i]) is not int or weights[i] < 0:
            raise Exception("weight values must be non-negative ints")
        K += weights[i]

    # Check if using constant weights or by-day weights
    if method is None:
        w_N = N
    elif method == "_constant":
        w_N = 1
    elif method == "_days":
        w_N = len(dat["dayLength"])
    else:
        raise Exception("method type " + method + " not supported")

    # Initialize weights to particular values (default 0)
    if E0 is not None:
        if type(E0) is not np.ndarray:
            raise Exception("E0 must be an array")

        if E0.shape == (w_N * K,):
            eInit = E0.copy()
        elif E0.shape == (w_N, K):
            eInit = E0.flatten()
        else:
            raise Exception("E0 must be shape (w_N*K,) or (w_N,K), not " +
                            str(E0.shape))
    else:
        eInit = np.zeros(w_N * K)

    # Do sanity checks on hyperparameters
    if "sigma" not in hyper:
        raise Exception("WARNING: sigma not specified in hyper dict")
    if "alpha" in hyper:
        raise Exception("WARNING: alpha is not supported")
    if method == "_constant":
        if "sigInit" not in hyper or hyper["sigInit"] is None:
            print("WARNING: sigInit being set to sigma for method", method)
    if method == "_days":
        if "sigDay" not in hyper or hyper["sigDay"] is None:
            print("WARNING: sigDay being set to sigma for method", method)

    # Get index of start of each day
    if ("dayLength" not in dat) and (
        ("sigDay" in hyper and hyper["sigDay"] is not None) or
        (method == "_days")):
        print("WARNING: sigDay has no effect, dayLength not supplied in dat")
        dat["dayLength"] = np.array([], dtype=int)

    # Account for missing trials from running xval (i.e. gaps from test set)
    if "missing_trials" in dat and dat["missing_trials"] is not None:
        if len(dat["missing_trials"]) != N:
            raise Exception("missing_trials must be length N if used")
    else:
        dat["missing_trials"] = None

    # -----
    # MAP estimate
    # -----

    # Prepare minimization of loss function, Memoize to preserve Jac+Hess info
    lossfun = memoize(negLogPost)
    my_args = (dat, hyper, weights, method)

    if showOpt:
        opts = {"disp": True}
        callback = print
    else:
        opts = {"disp": False}
        callback = None

    # Actual optimization call
    # Uses 'hessp' to pass a function that calculates product of Hessian
    #    with arbitrary vector
    if showOpt:
        print("Obtaining MAP estimate...")
    result = minimize(
        lossfun,
        eInit,
        jac=lossfun.jacobian,
        hessp=lossfun.hessian_prod,
        method="trust-ncg",
        tol=1e-9,
        args=my_args,
        options=opts,
        callback=callback,
    )

    # Recover the results of the optimization
    eMode = result.x
    # dict of sparse components of Hess
    Hess = lossfun.hessian(eMode, *my_args)

    # Print message if optimizer does not converge (usually still pretty good)
    if showOpt and not result.success:
        print("WARNING — MAP estimate: minimize() did not converge\n",
              result.message)
        print("NOTE: this is ususally irrelevant as the optimizer still finds "
              "a good solution. If you are concerned, run a check of the "
              "Hessian by setting showOpt >= 2")

    # Run DerivCheck & HessCheck at eMode (will run ShowOpt-1 distinct times)
    if showOpt >= 2:
        print("** Jacobian and Hessian Check **")
        for check in range(showOpt - 1):
            print("\nCheck", check + 1, ":")
            jacHessCheck(lossfun, eMode, *my_args)
            print("")

    # -----
    # Evidence (Marginal likelihood)
    # -----

    # Prior and likelihood at eMode, also recovering the associated wMode
    if showOpt:
        print("Calculating evd, first prior and likelihood at eMode...")
    pT, lT, wMode = getPosteriorTerms(eMode, *my_args)

    # Posterior term (with Laplace approx), calculating sparse log determinant
    if showOpt:
        print("Now the posterior with Laplace approx...")
    center = DT_X_D(Hess["ddlogprior"], Hess["K"]) + Hess["H"]
    logterm_post = (1 / 2) * sparse_logdet(center)

    # Compute Log evd and construct dict of likelihood, prior,
    #   and posterior terms
    logEvd = lT["logli"] + pT["logprior"] - logterm_post
    if showOpt:
        print("Evidence:", logEvd)

    # Package up important terms to return
    llstruct = {"lT": lT, "pT": pT, "eMode": eMode}

    return wMode, Hess, logEvd, llstruct


def negLogPost(*args):
    """Returns negative log posterior (and its first and second derivative)
    Intermediary function to allow for getPosteriorTerms to be optimized

    Args:
        same as getPosteriorTerms()

    Returns:
        negL : negative log-likelihood of the posterior
        dL : 1st derivative of the negative log-likelihood
        ddL : 2nd derivative of the negative log-likelihood,
            kept as a dict of sparse terms!
    """

    # Get prior and likelihood terms
    [priorTerms, liTerms, _] = getPosteriorTerms(*args)  # pylint: disable=no-value-for-parameter

    # Negative log posterior
    negL = -priorTerms["logprior"] - liTerms["logli"]
    dL = -priorTerms["dlogprior"] - liTerms["dlogli"]
    ddL = {"ddlogprior": priorTerms["ddlogprior"], **liTerms["ddlogli"]}

    return negL, dL, ddL


def getPosteriorTerms(E_flat, dat, hyper, weights, method=None):
    """Given a sequence of parameters formatted as an N*K matrix, calculates
    random-walk log priors & likelihoods and their derivatives

    Args:
        E_flat : array, the N*K epsilon parameters, flattened to a single
        vector
        ** all other args are same as in getMAP **

    Returns:
        priorTerms : dict, the log-prior as well as 1st + 2nd derivatives
        liTerms : dict, the log-likelihood as well as 1st + 2nd derivatives
        W : array, the weights, calculated directly from E_flat
    """

    # !!! TEMPORARY --- Need to update !!!
    if method in ["_days", "_constant"]:
        raise Exception(
            "Need efficient calculations for _constant or _days methods")

    # ---
    # Initialization
    # ---

    # If function is called directly instead of through getMAP,
    #       fill in dummy values
    if "dayLength" not in dat:
        dat["dayLength"] = np.array([], dtype=int)
    if "missing_trials" not in dat:
        dat["missing_trials"] = None

    # Unpack input into g
    g = read_input(dat, weights)
    N, K = g.shape

    # Determine type of analysis (standard, constant, or day weights)
    if method is None:
        w_N = N
        # the first trial index of each new day
        days = np.cumsum(dat["dayLength"], dtype=int)[:-1]
        missing_trials = dat["missing_trials"]
    elif method == "_constant":
        w_N = 1
        days = np.array([], dtype=int)
        missing_trials = None
    elif method == "_days":
        w_N = len(dat["dayLength"])
        days = np.arange(1, w_N, dtype=int)
        missing_trials = None
    else:
        raise Exception("method " + method + " not supported")

    # Check shape of epsilon, with
    #   w_N (effective # of trials) * K (# of weights) elements
    if E_flat.shape != (w_N * K,):
        print(E_flat.shape, w_N, K, method)
        raise Exception("parameter dimension mismatch (#trials * #weights)")

    # ---
    # Construct random-walk prior, calculate priorTerms
    # ---

    # Construct random walk covariance matrix Sigma^-1, use sparsity for speed
    invSigma = make_invSigma(hyper, days, missing_trials, w_N, K)

    # Calculate the log-determinant of prior covariance,
    #   the log-prior, 1st, & 2nd derivatives
    logdet_invSigma = np.sum(np.log(invSigma.diagonal()))
    logprior = (1 / 2) * (logdet_invSigma - E_flat @ invSigma @ E_flat)
    dlogprior = -invSigma @ E_flat
    ddlogprior = -invSigma

    priorTerms = {
        "logprior": logprior,
        "dlogprior": dlogprior,
        "ddlogprior": ddlogprior,
    }

    # ---
    # Construct likelihood, calculate liTerms
    # ---

    # Reconstruct actual weights from E values
    E = np.reshape(E_flat, (K, w_N), order="C")
    W = np.cumsum(E, axis=1)

    # Calculate probability of Right on each trial
    y = dat["y"] - 1
    gw = np.sum(g * W.T, axis=1)
    pR = 1 / (1 + np.exp(-gw))

    # Preliminary calculations for 1st and 2nd derivatives
    dlliList = g * (y - pR)[:, None]

    alpha = (pR**2 - pR)[:, None, None]
    HlliList = alpha * (g[:, :, None] @ g[:, None, :])

    # INSERT CODE HERE TO HANDLE _days OR _constant METHODS

    # Calculate the log-likelihood and 1st & 2nd derivatives
    logli = np.sum(y * gw - np.logaddexp(0, gw))
    dlogli = DTinv_v(dlliList.flatten("F"), K)
    ddlogli = {"H": myblk_diags(HlliList), "K": K}

    liTerms = {"logli": logli, "dlogli": dlogli, "ddlogli": ddlogli}

    return priorTerms, liTerms, W
