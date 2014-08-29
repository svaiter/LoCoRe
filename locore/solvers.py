import numpy as np
import scipy.linalg as lin

from .algorithms import forward_backward, admm, douglas_rachford
from .operators import soft_thresholding, dual_prox

def solve_l1_synthesis_lagrangian(Phi, y, la=1.0, maxiter=100):
    prox_f = lambda x, tau: soft_thresholding(x, la * tau)
    grad_g = lambda x: np.dot(Phi.T, np.dot(Phi, x) - y)
    n = Phi.shape[1]
    L = lin.norm(Phi, 2) ** 2
    x = forward_backward(prox_f, grad_g, np.zeros((n,1)), L,
                         maxiter=maxiter, method='fista',
                         full_output=0, retall=0)
    return x

def solve_l1_synthesis_constrained(Phi, y, maxiter=100):
    n = Phi.shape[1]
    prox_f = soft_thresholding
    prox_g = lambda x, tau: x + np.dot(Phi.T, np.linalg.solve(np.dot(Phi,Phi.T),
    y - np.dot(Phi,x)))
    return douglas_rachford(prox_f, prox_g, np.zeros((n,1)),
                            maxiter=1000, full_output=0, retall=0)

def solve_l1_analysis(D, Phi, y, la=1.0, maxiter=100, constrained=False):
    n, p = D.shape
    q = Phi.shape[0]
    prox_g = lambda u, tau: u
    if constrained:
        prox_q = lambda u, tau: y
        prox_r = lambda u, tau: soft_thresholding(u, tau)
    else:
        prox_q = lambda u, tau: lin.lstsq(((1.0+tau)*np.eye(q)), (u+tau*y))[0]
        prox_r = lambda u, tau: soft_thresholding(u, la*tau)
    prox_f = lambda u, tau: np.concatenate((
        prox_q(u[:q,:], tau),
        prox_r(u[q:,:], tau)))
    prox_fs = dual_prox(prox_f)
    OpPD = np.concatenate((Phi, D.T))
    K = lambda u: np.dot(OpPD, u)
    K.T = lambda u: np.dot(OpPD.T, u)
    tau = 0.99/np.sqrt(lin.norm(Phi) ** 2 + lin.norm(D.T) ** 2)
    sigma = tau
    return admm(prox_fs, prox_g, K, np.random.randn(n,1),
                sigma=sigma, tau=tau, maxiter=maxiter,
                full_output=0, retall=0)
