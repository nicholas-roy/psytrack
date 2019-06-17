import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from .getMAP import getMAP, getPosteriorTerms
from psytrack.aux.auxFunctions import (
    DTinv_v,
    Dinv_v,
    DTv,
    Dv,
    DT_X_D,
    make_invSigma,
    sparse_logdet,
)


def hyperOpt(dat, hyper, weights, optList, method=None, showOpt=0, jump=2):
    """Given data and set of hyperparameters, uses decoupled Laplace to find the
    optimal hyperparameter values (i.e. the sigmas that maximize evidence)

    Args:
        dat : dict, all data from a specific subject
        hyper : dict, hyperparameters and initial values used to construct prior
            Must at least include sigma, can also include sigInit, sigDay
        weights : dict, name and count of which weights in dat['inputs'] to fit
        optList : list, hyperparameters in 'hyper' to be optimized
        method : str, control over type of fit,
            None is standard, '_days' and '_constant' also supported
        showOpt : int, 0 : no text, 1 : verbose,
            2+ : Hess + deriv check, done showOpt-1 times
        jump : int, how many times the alg can find suboptimal evd before quit

    Returns:
        best_hyper : hyperparameter values that maximizes evidence of data
        best_logEvd : log-evidence associated with optimal hyperparameter values
        best_wMode : the MAP weights found using best_hyper, maximizing logEvd
        best_Hess : dict, the Hessian found using best_hyper, maximizing logEvd
    """

    # Initialization of optimization
    opt_keywords = {
        "dat": dat,
        "hyper": hyper,
        "weights": weights,
        "optList": optList,
        "method": method,
    }

    current_hyper = hyper.copy()
    best_logEvd = None

    # Make sure all hyperparameters to be optimized are actually provided
    for val in optList:
        if (val not in hyper) or (hyper[val] is None):
            raise Exception("cannot optimize a hyperparameter not given")

    # -----
    # Hyperparameter Optimization
    # -----

    current_jump = jump
    while True:

        if best_logEvd is None:
            E0 = None
        else:
            E0 = llstruct["eMode"]  # pylint: disable=used-before-assignment

        # First get MAP for initial hyperparameter setting
        wMode, Hess, logEvd, llstruct = getMAP(
            dat,
            current_hyper,
            weights,
            E0=E0,
            method=method,
            showOpt=int(showOpt > 1))

        # Update best variables
        if best_logEvd is None:
            best_logEvd = logEvd
        if logEvd >= best_logEvd:
            current_jump = jump
            best_hyper = current_hyper.copy()
            best_logEvd = logEvd
            best_wMode = wMode
            best_Hess = Hess
        else:
            # If a worse logEvd found, reduce jump by one and
            # move hypers to midpoints, keep old bests
            current_jump -= 1
            for val in optList:
                current_hyper.update({
                    val: (current_hyper[val] + best_hyper[val]) / 2
                })

        if showOpt:
            print("\nInitial evidence:", np.round(logEvd, 5))
            for val in optList:
                print(val, np.round(np.log2(current_hyper[val]), 4))

        # Jump to end if evidence was worse enough times
        if not current_jump:
            if showOpt:
                print("Jumping to end due to no improvement in evidence")
            break

        # Now decouple prior terms from likelihood terms and store values
        K = llstruct["lT"]["ddlogli"]["K"]
        H = llstruct["lT"]["ddlogli"]["H"]
        ddlogprior = llstruct["pT"]["ddlogprior"]
        eMode = llstruct["eMode"]

        LL_v = DTinv_v(H @ Dinv_v(eMode, K), K) + ddlogprior @ eMode

        opt_keywords.update({
            "LL_terms": llstruct["lT"]["ddlogli"],
            "LL_v": LL_v
        })

        # Optimize over hyperparameters
        if showOpt:
            print("\nStarting optimization...")
            opts = {"maxiter": 15, "disp": True}
            callback = print
        else:
            opts = {"maxiter": 15, "disp": False}
            callback = None

        # Do hyperparameter optimization in log2
        optVals = []
        for val in optList:
            if np.isscalar(current_hyper[val]):
                optVals += [np.log2(current_hyper[val])]
            else:
                optVals += np.log2(current_hyper[val]).tolist()

        result = minimize(
            hyperOpt_lossfun,
            optVals,
            args=opt_keywords,
            method="BFGS",
            options=opts,
            callback=callback,
        )

        if showOpt:
            print("\nRecovered evidence:", np.round(-result.fun, 5))

        count = 0
        for val in optList:
            if np.isscalar(current_hyper[val]):
                current_hyper.update({val: 2**result.x[count]})
                count += 1
            else:
                current_hyper.update({val: 2**result.x[count:count + K]})
                count += K
            if showOpt:
                print(val, np.round(np.log2(current_hyper[val]), 4))

        # Test to see if hyperparameters have convereged
        diff = np.linalg.norm((optVals - result.x) / optVals)
        if showOpt:
            print("\nDifference:", np.round(diff, 4))
        if diff < 0.1:
            break

    # # If hyperparameters converged, recalculate evidence and wMode
    # # If evidence is not improving, no need to recalculate, jump to end
    # if current_jump:
    #     # Calculate true evidence of final set of hyperopt
    #     wMode, Hess, logEvd, llstruct = getMAP_PBups(dat,current_hyper,weights,
    #                                   method=method, showOpt=int(showOpt>1))

    #     if logEvd >= best_logEvd:
    #         best_hyper = current_hyper.copy()
    #         best_logEvd = logEvd
    #         best_wMode = wMode

    if showOpt:
        print("Coverged! Final evidence:", np.round(best_logEvd, 5))
        for val in optList:
            print(val, np.round(np.log2(best_hyper[val]), 4))

    return best_hyper, best_logEvd, best_wMode, best_Hess


def hyperOpt_lossfun(optVals, keywords):
    """Loss function used by decoupled Laplace to optimize for evidence over
    changes in hyperparameters

    Args:
        optVals : hyperparameter values currently caluclating approximate
            evidence for corresponds to hyperparameters listed in OptList
        keywords : dictionary of other values needed for optimization

    Returns:
        evd : the negative evidence (to be minimized)
    """

    # Recover N & K
    N = keywords["dat"]["y"].shape[0]
    K = keywords["LL_terms"]["K"]
    method = keywords["method"]
    dat = keywords["dat"]
    weights = keywords["weights"]

    # Reconstruct the prior covariance
    hyper = keywords["hyper"].copy()

    count = 0
    for val in keywords["optList"]:
        if np.isscalar(hyper[val]):
            hyper.update({val: 2**optVals[count]})
            count += 1
        else:
            hyper.update({val: 2**optVals[count:count + K]})
            count += K

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

    invSigma = make_invSigma(hyper, days, missing_trials, w_N, K)
    ddlogprior = -invSigma

    # Retrieve terms for decoupled Laplace appx.
    H = keywords["LL_terms"]["H"]
    LL_v = keywords["LL_v"]

    # Decoupled Laplace appx to new epsilon given new sigma
    DL_1 = DTv(LL_v, K)
    DL_2 = DT_X_D(ddlogprior, K)
    DL_3 = spsolve(DL_2 + H, DL_1)
    E_flat = Dv(DL_3, K)

    # Calculate likelihood and prior terms with new epsilon
    pT, lT, _ = getPosteriorTerms(
        E_flat, hyper=hyper, method=method, dat=dat, weights=weights)

    # Calculate posterior term, then approximate evidence for new sigma
    center = DL_2 + lT["ddlogli"]["H"]
    logterm_post = (1 / 2) * sparse_logdet(center)

    evd = pT["logprior"] + lT["logli"] - logterm_post

    return -evd
