import numpy as np
from scipy.sparse import csr_matrix, isspmatrix, diags, block_diag
from scipy.sparse.linalg import inv
from .auxFunctions import DT_X_D

def getCredibleInterval(Hess):
    return np.sqrt(invDiagHess(Hess)).reshape(Hess['K'],-1)

def invDiagHess(Hess):
    """
    True Hessian in e space: ddlogprior + DT^{-1} @ H @ D^{-1}
    Hessian in w space:  DT @ ddlogprior @ D + H, a block tridiagonal

    Args:
        Hess : dict, contains components needed to construct full Hessian

    Returns:
        invHess : array, the diagonal of the inverted negative Hessian
    """

    # Construct center
    center = -(DT_X_D(Hess["ddlogprior"], Hess["K"]) + Hess["H"])

    # Rearrange matrix such that it's blocked by K, not by N
    K = Hess["K"]
    N = int(Hess["ddlogprior"].shape[0] / K)
    ii = (np.reshape(np.arange(K * N), (N, -1),
                     order="F").T).flatten(order="F")
    M = center[ii]
    M = M[:, ii]

    # Do efficient inverse of block tridiagonal center
    vdiag, _, _ = invBlkTriDiag(M, K)

    # Reorder matrix back into blocked by N, rather than by K
    kk = np.argsort(ii)
    invHess = vdiag[kk]

    return invHess


def invBlkTriDiag(M, nn):
    """
    
    Efficiently inverts a block tridiagonal (a block diagonal matrix 
    with off-diagonal blocks) matrix. Blocks are (nn,nn) with nblocks
    equal to the # of blocks.
    
    Args:
        M : *sparse* square 2D array, the block tridiagonal matrix
        nn : size of each block

    Returns:
        MinvDiag : values on the diagonal of inverted M
        MinvBlocks : values on the main diagonal blocks of invM
        MinvBelowDiagBlocks : values on the off diagonal blocks of invM
    """
    if not isspmatrix(M):
        print("Casting M to sparse format")
        M = csr_matrix(M)

    nblocks = int(M.shape[0] / nn)  # number of total blocks

    # Matrices to store during recursions
    A = np.zeros((nn, nn, nblocks))  # for below-diagonal blocks
    B = np.zeros((nn, nn, nblocks))  # for diagonal blocks
    C = np.zeros((nn, nn, nblocks))  # for above-diagonal blocks
    D = np.zeros((nn, nn, nblocks))  # quantity to compute
    E = np.zeros((nn, nn, nblocks))  # quantity to compute

    # Initialize first D block
    inds = np.arange(nn)  # indices for 1st block
    B[:, :, 0] = M[np.ix_(inds, inds)].todense()
    C[:, :, 0] = M[np.ix_(inds, inds + nn)].todense()
    D[:, :, 0] = np.linalg.solve(B[:, :, 0], C[:, :, 0])

    # Initialize last E block
    inds = (nblocks - 1) * nn + inds  # indices for last block
    A[:, :, -1] = M[np.ix_(inds, inds - nn)].todense()
    B[:, :, -1] = M[np.ix_(inds, inds)].todense()
    E[:, :, -1] = np.linalg.solve(B[:, :, -1], A[:, :, -1])

    # Extract blocks A, B, and C
    for ii in np.arange(1, nblocks - 1):
        inds = np.arange(nn) + ii * nn  # indices for center block
        A[:, :, ii] = M[np.ix_(inds,
                               inds - nn)].todense()  # below-diagonal block
        B[:, :, ii] = M[np.ix_(inds, inds)].todense()  # middle diagonal block
        C[:, :, ii] = M[np.ix_(inds,
                               inds + nn)].todense()  # above diagonal block

    # Make a pass through data to compute D and E
    for ii in np.arange(1, nblocks - 1):
        # Forward recursion
        D[:, :, ii] = np.linalg.solve(
            B[:, :, ii] - A[:, :, ii] @ D[:, :, ii - 1], C[:, :, ii])

        # Backward recursion
        jj = nblocks - ii - 1
        E[:, :, jj] = np.linalg.solve(
            B[:, :, jj] - C[:, :, jj] @ E[:, :, jj + 1], A[:, :, jj])

    # Now form blocks of inverse covariance
    I = np.eye(nn)
    MinvBlocks = np.zeros((nn, nn, nblocks))
    MinvBelowDiagBlocks = np.zeros((nn, nn, nblocks - 1))
    MinvBlocks[:, :, 0] = np.linalg.inv(
        B[:, :, 0] @ (I - D[:, :, 0] @ E[:, :, 1]))
    MinvBlocks[:, :, -1] = np.linalg.inv(B[:, :, -1] -
                                         A[:, :, -1] @ D[:, :, -2])
    for ii in np.arange(1, nblocks - 1):
        # Compute diagonal blocks of inverse
        MinvBlocks[:, :, ii] = np.linalg.inv(
            (B[:, :, ii] - A[:, :, ii] @ D[:, :, ii - 1])
            @ (I - D[:, :, ii] @ E[:, :, ii + 1]))
        # Compute below-diagonal blocks
        MinvBelowDiagBlocks[:, :, ii -
                            1] = -D[:, :, ii - 1] @ MinvBlocks[:, :, ii]

    MinvBelowDiagBlocks[:, :, -1] = -D[:, :, -2] @ MinvBlocks[:, :, -1]

    # Extract just the diagonal elements
    MinvDiag = np.zeros(nn * nblocks)
    for ii in np.arange(nblocks):
        MinvDiag[ii * nn:(ii + 1) * nn] = np.diag(MinvBlocks[:, :, ii])

    return MinvDiag, MinvBlocks, MinvBelowDiagBlocks
