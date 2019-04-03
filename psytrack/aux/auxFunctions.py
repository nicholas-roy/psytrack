from scipy.sparse.linalg import splu
from scipy.sparse import isspmatrix_csc, diags
import numpy as np


def myblk_diags(A):
    """
	Function to take (N,K,K) matrix A and put each len(N) entry of A[:,i,j] as
	the diagonal of an (N*K x N*K) matrix. For speed, sparsity is used — define
	the diagonal values of the final matrix then use diags to construct the
	final matrix. Some minor Python black magic is used below, primarily the use
	of negative indices to index from the end, sorry.
    """
    # Retrieve shape of given matrix
    N, K, _ = np.shape(A)

    # Will need (2K-1) diagonals, with the longest N*K long
    d = np.zeros((2 * K - 1, N * K))

    # Need to keep track of each diagonal's offset in the final
    # 	matrix : (0,1,...,K-1,-K+1,...,-1)
    offsets = np.hstack((np.arange(K), np.arange(-K + 1, 0))) * N

    # Retrieve diagonal values from A to fill in d
    for i in range(K):
        for j in range(K):
            m = np.min([i, j])
            d[j - i, m * N:(m + 1) * N] = A[:, i, j]

    # After diagonals are constructed, use sparse function diags() to make
    # 	matrix, then blow up to full size
    return diags(d, offsets, shape=(N * K, N * K), format="csc")


def sparse_logdet(A):
    """
    Calculate the log determinant using sparse LU decomposition, product of 
    diagonal values of L and U matrix give determinant up to a sign change.
    Know positive determinant, so take log of the absolute value of the 
    diagonal values and sum them for log-determinant.

    See link for math and implementation details:
    http://stackoverflow.com/a/19616987  (first comment w/ Wikipedia link)
    """
    if not isspmatrix_csc(A):  # needed for splu() decomposition to work
        raise Exception(
            "sparse_logdet: matrix passed is not in sparse csc form")

    aux = splu(A)
    return np.sum(
        np.log(np.abs(aux.L.diagonal())) + np.log(np.abs(aux.U.diagonal())))


def make_invSigma(hyper, days, missing_trials, N, K):
    """Returns the banded prior matrix
    
    Constructs the random-walk prior, accounting for new sessions and trials
    omitted for cross-validation
    
    Args:
        hyper (dict): hyperparameters and values
        days (array): indices of trials that start a session
        missing_trials (array): indices of trials removed for cross-validation
        N (int): number of trials
        K (int): number of weights
    
    Returns:
        (sparse array): prior matrix
    """

    # Note: setting a sigma value at index i, adjusts the change between trials i-1 and i
    sigma = hyper["sigma"]

    if "sigInit" in hyper and hyper["sigInit"] is not None:
        sigInit = hyper["sigInit"]
    else:
        sigInit = sigma

    if "sigDay" in hyper and hyper["sigDay"] is not None:
        sigDay = hyper["sigDay"]
    else:
        sigDay = sigma

    if np.isscalar(sigma):
        invSigma_flat_k = np.ones(N) * sigma**2
        invSigma_flat_k[days] = sigDay**2  # add sigDay to day changes
        invSigma_flat_k[0] = sigInit**2  # add sigInit to beginning

        if missing_trials is not None:  # add extra sigma variance if a test trial gap
            invSigma_flat_k += missing_trials * sigma**2

        invSigma_flat = np.tile(invSigma_flat_k, K)
        return diags(invSigma_flat**-1)

    elif type(sigma) in [np.ndarray, list]:
        if len(sigma) != K:
            raise Exception("number of sigmas is not K")

        invSigma_flat = np.zeros(N * K)
        for k in range(K):
            invSigma_flat[k * N:(k + 1) * N] = sigma[k]**2

            if np.isscalar(sigDay):
                invSigma_flat[k * N + days] = sigDay**2
            else:
                invSigma_flat[k * N + days] = sigDay[k]**2

            if np.isscalar(sigInit):
                invSigma_flat[k * N] = sigInit**2
            else:
                invSigma_flat[k * N] = sigInit[k]**2

            if (missing_trials is
                    not None):  # add extra sigma variance if a test trial gap
                invSigma_flat[k * N:(k + 1) *
                              N] += missing_trials * sigma[k]**2

        return diags(invSigma_flat**-1)

    else:
        raise Exception("sigma must be of appropriate type, not" +
                        str(type(sigma)))


def read_input(D, w):
    """Creates carrier vector of inputs g
    
    From dataset D and dict of weights and counts w, collect all the inputs
    into a single K x N matrix g
    
    Args:
        D (dict): standard dataset
        w (dict): count of each type of weight to include
    
    Returns:
        g (array): matrix of inputs to model
    """

    # Determine dimensions N and K, create g
    N = len(D["y"])

    K = 0
    for i in w.keys():
        K += w[i]

    g = np.zeros((N, K))
    g_ind = 0

    for i in sorted(w.keys()):
        if i == "bias":
            g[:, g_ind:g_ind + 1] = 1
        else:
            try:
                g[:, g_ind:g_ind + w[i]] = D["inputs"][i][:, :w[i]]
            except:
                raise Exception(
                    str(i) + " given in weights not in dataset inputs")

        g_ind += w[i]

    return g


def trim(dat, START=0, END=0):
    """Utility function for slicing a dataset with a start / end point
    
    Returns a standard data set that has been sliced according to START and END
    Especially inportant for keeping session info intact
    
    Args:
        dat (dict): a standard dataset
        START (int): The trial where the new dataset should start, 0 is start
        END (int): The trial where the new dataset should end, 0 is end, can
            also take negative values
    
    Returns:
        new_dat (dict): the trimmed dataset
    """

    if (not START) and (not END):
        return dat

    N = len(dat["y"])

    if START < 0:
        START = N + START
    if START > N:
        raise Exception("START > N : " + str(START) + ", " + str(N))
    if END <= 0:
        END = N + END
    if END > N:
        END = N
    if START >= END:
        raise Exception("START >= END : " + str(START) + ", " + str(END))

    new_dat = {}
    for k in dat.keys():

        if k == "inputs":
            continue

        try:
            if N == dat[k].shape[0]:
                new_dat[k] = dat[k][START:END]
            else:
                new_dat[k] = dat[k].copy()
        except:
            new_dat[k] = dat[k]

    inputs = {}
    for i in dat["inputs"].keys():
        inputs[i] = dat["inputs"][i][START:END]
    new_dat["inputs"] = inputs

    if "dayLength" in new_dat and new_dat["dayLength"].size:
        cumdays = np.cumsum(new_dat["dayLength"])
        min_id = np.where(cumdays > START)[0][0]
        max_id = np.where(cumdays < END)[0][-1] + 1
        new = new_dat["dayLength"][min_id:max_id + 1].copy()
        new[0] = cumdays[min_id] - START
        new[-1] = END - cumdays[max_id - 1]
        if len(new) == 1:
            new[0] = END - START
        new_dat["dayLength"] = new

    new_dat["skimmed"] = {"START": START, "END": END}

    return new_dat


def DT_X_D(ddlogprior, K):
    """
    Computes D.T @ ddlogprior @ D, where D is the blocked 
    difference matrix much more quickly
    """
    dd = ddlogprior.diagonal().reshape((K, -1)).copy()

    main_diag = dd.copy()
    main_diag[:, :-1] += main_diag[:, 1:]
    main_diag = main_diag.flatten()

    off_diags = dd.copy()
    off_diags[:, 0] = 0
    off_diags = -off_diags.flatten()[1:]

    NK = main_diag.shape[0]
    A = np.zeros((3, NK))
    A[0] = main_diag
    A[1, :-1] = off_diags
    A[2, :-1] = off_diags

    return diags(A, [0, -1, 1], shape=(NK, NK), format="csc")


def Dv(v, K):
    """
    Computes D @ v, where D is the blocked difference matrix much more quickly
    """
    v2 = v.reshape(K, -1)
    v3 = np.hstack((v2[:, 0:1], np.diff(v2, axis=1)))
    v4 = v3.flatten()
    return v4


def DTv(v, K):
    """
    Computes D.T @ v, where D is the blocked difference matrix much more quickly
    """
    v2 = np.flip(v.reshape(K, -1), axis=1)
    v3 = np.hstack((v2[:, 0:1], np.diff(v2, axis=1)))
    v4 = np.flip(v3, axis=1).flatten()
    return v4


def Dinv_v(v, K):
    """
    Computes D^-1 @ v, where D is the blocked difference matrix
    much more quickly
    """
    v2 = v.reshape(K, -1)
    v3 = np.cumsum(v2, axis=1).flatten()
    return v3


def DTinv_v(v, K):
    """
    Computes D^-T @ v, where D is the blocked difference matrix much more quickly
    """
    v2 = np.flip(v.reshape(K, -1), axis=1)
    v3 = np.flip(np.cumsum(v2, axis=1), axis=1).flatten()
    return v3
