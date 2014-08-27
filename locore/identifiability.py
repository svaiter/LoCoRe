from __future__ import division
import numpy as np
import scipy as sp

from .operators import (
    dual_prox,
    linf_l2_norm,
    l1_support,
    l1_l2_support,
    l1_l2_normalize_subvector
)
from .algorithms import douglas_rachford
from .utils import null, l1ball_projection


# L^1 synthesis
def crit_l1_synthesis(Phi, x):
    I = l1_support(x)
    J = ~I
    sI = np.sign(x)[I, :]
    corr = np.dot(np.dot(Phi[:, J].T, sp.linalg.pinv(Phi[:, I].T)), sI)
    ic = np.max(np.abs(corr))
    res = {
        'ic': ic,
        'I': I,
        'J': J,
        'sI': sI,
        'corr': corr
    }
    return res


def crit_l1_analysis(D, Phi, x, maxiter=50):
    # Dimensions of the problem
    N = np.size(D, 0)
    P = np.size(D, 1)
    Q = np.size(Phi, 0)

    # Generate sub-dict of given cosparsity
    I = (np.abs(np.dot(D.T, x)) > 1e-5).flatten()
    J = ~I
    DI = D[:, I]
    DJ = D[:, J]

    # Compute operators involved in criterions
    U = null(DJ.T)
    gram = np.dot(Phi.T, Phi)
    inside = np.dot(np.dot(U.T, gram), U)
    if np.prod(inside.shape) != 0:
        inside = sp.linalg.pinv(inside)
    A = np.dot(np.dot(U, inside), U.T)
    Omega = np.dot(sp.linalg.pinv(DJ),
                   np.dot((np.eye(N) - np.dot(gram, A)), DI))

    # D-sign
    ds = np.sign(np.dot(DI.T, x))

    # Compute IC
    if np.prod(null(DJ).shape) != 0:
        # Returns the solution of the problem
        #   min_{w \in Ker(D_J)} ||Omega sign(D_I^* x_0) - w||_\infty
        Omega_x0 = np.dot(Omega, s)
        proj = np.dot(np.dot(DJ.T, sp.linalg.pinv(np.dot(DJ, DJ.T))), DJ)
        prox_indic = lambda w, la: w - np.dot(proj, w)

        proxinf = dual_prox(l1ball_projection)
        prox_obj = lambda x, la: -proxinf(Omega_x0 - x, la) + Omega_x0

        w = douglas_rachford(prox_indic, prox_obj,
                             np.zeros((np.size(DJ, 1), 1)), maxiter=maxiter)
        ic = np.max(np.abs(Omega_x0 - w))
    else:
        ic = np.max(np.abs(np.dot(Omega, ds)))

    res = {
        'ic': ic,
        'I': I,
        'J': J,
        'U': U,
        'A': A,
        'Omega': Omega,
        'ds': ds
    }

    return res


def crit_l1_l2_synthesis(blocks, Phi, x):
    I = l1_l2_support(blocks, x)
    J = ~I
    e = l1_l2_normalize_subvector(blocks, I, x)
    corr = np.dot(np.dot(Phi[:, J].T, sp.linalg.pinv(Phi[:, I].T)), e)
    corr_extended = np.zeros(x.shape)
    corr_extended[I, :] = corr[I, :]
    ic = linf_l2_norm(blocks, corr_extended)
    res = {
        'ic': ic,
        'I': I,
        'J': J,
        'e': e,
        'corr': corr
    }
    return res


def crit_nuclear(Phi, x):
    n1, n2 = x.shape
    n = n1 * n2

    resh = lambda u: np.reshape(u, (n1, n2))
    eps = 1e-8

    # SVD of x: x=U*S*V'
    U, S, V = sp.linalg.svd(x)
    S = np.diag(S)
    r = np.sum(S > eps)

    # span orthogonal to row and col spaces
    U0 = U[:, r:]
    V0 = V[:, r:]
    # space row and col spaces
    U = U[:, :r]
    V = V[:, :r]

    # projector on S
    #   P_S(Y) = U0*U0'*Y*V0*V0'
    Vg = np.dot(V0, V0.T)
    Ug = np.dot(U0, U0.T)
    PS = np.kron(Vg, Ug)
    PSmat = lambda y: np.dot(Ug, np.dot(y, Vg))

    # T = {X ; U0*X*V0' = 0}
    #   = Im(f) = { f(A,B) = U*A + B*V' ; A in R^{r x n2}, B in R^{n1 x r} }
    PT = np.eye(n) - PS
    # basis of T
    B = np.concatenate((np.kron(np.eye(n2), U), np.kron(V, np.eye(n1))),
                       axis=1)

    PhiT = np.dot(Phi, PT)
    PhiB = np.dot(Phi, B)
    # PhiT = Phi - PhiS

    # generalized sign
    e = np.dot(U, V.T)
    e = e.reshape((n, 1))

    rPhiT = np.linalg.matrix_rank(PhiB)
    # dim(T) = n1^2 - (n1-r)^2   (for square matrices)
    # dimT = rank(PT);
    dimT = n1 ** 2 - (n1-r) ** 2

    if rPhiT < dimT:
        return np.Inf
    elif rPhiT == dimT:
        # IC(X) = | PhiS' * PhiT^{+,*} * e |_{*,inf}
        # ic = norm( resh( PhiS' * pinv(PhiT)' * e )  );

        # Compute pinv(PhiT)' * e
        #    = Phi * B * pinv( B'*Phi'*Phi*B ) * B' * e(:)
        v = np.dot(PhiB, np.dot(sp.linalg.pinv(np.dot(PhiB.T, PhiB)),
                                np.dot(B.T, e)))
        ic = sp.linalg.norm(PSmat(resh(np.dot(Phi.T, v))))
        return ic
    else:
        raise Exception("injectivity problem")
