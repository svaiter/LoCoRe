"""
The :mod:`locore.utils` module includes common functions to LoCoRe.
"""

from __future__ import division
import numpy as np
import scipy
from scipy import linalg as lin

def null(A, tol=1e-10):
    (m,n) = A.shape
    U, S, Vh = linalg.svd(A)
    if m > 1:
        s = np.diag(S)
    elif m == 1:
        s = S[1]
    else:
        s = 0
    r = np.sum(s > tol)
    return Vh.T[:,r:n]


def operator_norm(linop, n=None, maxiter=30, check=False):
    if hasattr(linop, 'norm') and not check:
        return linop.norm
    if n is None:
        n = np.random.randn(linop.dim[1], 1)
    if np.size(n) == 1:
        u = np.random.randn(n, 1)
    else:
        u = n
    unorm = lin.norm(u)
    if unorm > 1e-10:
        u = u / unorm
    else:
        return 0
    e = []
    for i in range(maxiter):
        if hasattr(linop, 'T'):
            v = linop.T(linop(u))
        else:
            # assume square (implicit) operator
            v = linop(u)
        e.append((u[:] * v[:]).sum())
        vnorm = lin.norm(v[:])
        if vnorm > 1e-10:
            u = v / vnorm
        else:
            return 0
    L = e[-1]
    return L

# Adrien Gaidon - INRIA - 2011
def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.

    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
        # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

# Adrien Gaidon - INRIA - 2011
def l1ball_projection(v, s=1):
    """ Compute the Euclidean projection on a L1-ball

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the L1-ball

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s

    Notes
    -----
    Solves the problem by a reduction to the positive simplex case

    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    if len(v.shape) > 1:
        n,m = v.shape
        assert m == 1, "Must be 1D vector"
        v = v.flatten()
    else:
        n, = v.shape  # will raise ValueError if v is not 1-D
        m = 0
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
        # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    if m <> 0:
        return w[:,np.newaxis]
    else:
        return w
