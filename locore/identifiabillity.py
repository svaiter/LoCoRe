from __future__ import division
import numpy as np
import scipy

from .operators import dual_prox, linf_l2_norm, l1_support, l1_l2_support, l1_l2_normalize_subvector
from .algorithms import douglas_rachford
from .utils import null, l1ball_projection

# L^1 synthesis
def crit_l1_synthesis(Phi, x):
    I = l1_support(x)
    J = ~I
    sI = np.sign(x)[I,:]
    corr = np.dot(np.dot(Phi[:,J].T, scipy.linalg.pinv(Phi[:,I].T)), sI)
    ic = np.max(np.abs(corr))
    res = {
        'ic' : ic,
        'I' : I,
        'J' : J,
        'sI' : sI,
        'corr' : corr
    }
    return res

# L^1 analysis
def _ic_minimization(DJ, Omega_x0, maxiter):
    """
    Returns the solution of the problem
        min_{w \in Ker(D_J)} ||Omega sign(D_I^* x_0) - w||_\infty
    """
    proj = np.dot(np.dot(DJ.T,scipy.linalg.pinv(np.dot(DJ,DJ.T))), DJ)
    prox_indic = lambda w, la: w - np.dot(proj, w)

    proxinf = dual_prox(l1ball_projection)
    prox_obj = lambda x, la: -proxinf(Omega_x0 - x, la) + Omega_x0

    w = douglas_rachford(prox_indic, prox_obj, np.zeros((np.size(DJ,1),1)),
        maxiter=maxiter)
    return np.max(np.abs(Omega_x0 - w))


def crit_l1_analysis(D, Phi, x):
    # Dimensions of the problem
    N = np.size(D, 0)
    P = np.size(D, 1)
    Q = np.size(Phi, 0)

    # Generate sub-dict of given cosparsity
    I = (np.abs(np.dot(D.T, x)) > 1e-5).flatten()
    J = ~I
    DI = D[:,I]
    DJ = D[:,J]

    # Compute operators involved in criterions
    U = null(DJ.T)
    gram = np.dot(Phi.T, Phi)
    inside = np.dot(np.dot(U.T, gram), U)
    if np.prod(inside.shape) <> 0:
        inside = scipy.linalg.pinv(inside)
    A = np.dot(np.dot(U, inside), U.T)
    Omega = np.dot(scipy.linalg.pinv(DJ), np.dot((np.eye(N) -  np.dot(gram,
        A)), DI))

    # Compute wRC
    wRC = scipy.linalg.norm(Omega, np.inf)

    # D-sign
    ds = np.sign(np.dot(DI.T, x))

    # Compute IC-noker
    ic_noker = lambda s : np.max(np.abs(np.dot(Omega,s)))
    ICnoker = ic_noker(ds)

    # Compute IC
    if np.prod(null(DJ).shape) <> 0:
        ic_ker = lambda s: _ic_minimization(DJ, np.dot(Omega, s), maxiter)
    else:
        ic_ker = ic_noker

    ic = ic_ker(ds)

    res = {
        'wRC' : wRC,
        'IC_noker' : ICnoker,
        'ic' : ic,
        'ic_noker' : ic_noker,
        'ic_ker' : ic_ker,
        'I' : I,
        'J' : J,
        'U' : U,
        'A' : A,
        'Omega' : Omega,
        'ds' : ds
    }

    return res

def crit_l1_l2_synthesis(blocks, Phi, x):
    I = l1_l2_support(blocks, x)
    J = ~I
    e = l1_l2_normalize_subvector(blocks, I, x)
    corr = np.dot(np.dot(Phi[:,J].T, scipy.linalg.pinv(Phi[:,I].T)), e)
    corr_extended = np.zeros(x.shape)
    corr_extended[I,:] = corr[I,:]
    ic = linf_l2_norm(blocks, corr_extended)
    res = {
        'ic' : ic,
        'I' : I,
        'J' : J,
        'e' : e,
        'corr' : corr
    }
    return res
