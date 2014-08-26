from __future__ import division
from numpy.testing import assert_array_almost_equal

import numpy as np
import scipy.linalg as lin
from locore.algorithms import admm, douglas_rachford, forward_backward, forward_backward_dual
from locore.operators import soft_thresholding

methods = ['fb', 'fista', 'nesterov']


def test_dr_virtual_zero():
    # Virtual 0-prox
    prox_f = lambda u, la: 0 * 0
    prox_g = lambda u, la: u * 0

    # observations of size (5,1)
    y = np.zeros((5, 1))
    x_rec = douglas_rachford(prox_f, prox_g, y)
    assert_array_almost_equal(y, x_rec)

    # observations of size (5,2)
    y = np.zeros((5, 2))
    x_rec = douglas_rachford(prox_f, prox_g, y)
    assert_array_almost_equal(y, x_rec)


def test_dr_zero():
    # Prox of F, G = 0
    prox_f = lambda u, la: u
    prox_g = lambda u, la: u

    # observations of size (5,1)
    y = np.zeros((5, 1))
    x_rec = douglas_rachford(prox_f, prox_g, y)
    assert_array_almost_equal(y, x_rec)

    # observations of size (5,2)
    y = np.zeros((5, 2))
    x_rec = douglas_rachford(prox_f, prox_g, y)
    assert_array_almost_equal(y, x_rec)


def test_dr_l1_cs():
    # Dimension of the problem
    n = 200
    p = n // 4

    # Matrix and observations
    A = np.random.randn(p, n)
    # Use a very sparse vector for the test
    x = np.zeros((n, 1))
    x[1, :] = 1
    y = np.dot(A, x)

    # operator callbacks
    prox_f = soft_thresholding
    prox_g = lambda x, tau: x + np.dot(A.T, lin.solve(np.dot(A, A.T),
        y - np.dot(A, x)))

    x_rec = douglas_rachford(prox_f, prox_g, np.zeros((n, 1)), maxiter=1000)
    assert_array_almost_equal(x, x_rec)


def test_admm_virtual_zero():
    # Virtual 0-prox
    prox_fs = lambda u, la: u * 0
    prox_g = lambda u, la: u * 0

    # ndarray
    k_nd = np.zeros((5, 5))
    # explicit
    k_exp = lambda u: 0 * u
    k_exp.T = lambda u: 0 * u

    # observations of size (5,1)
    y = np.zeros((5, 1))
    x_rec = admm(prox_fs, prox_g, k_nd, y)
    assert_array_almost_equal(y, x_rec)
    x_rec = admm(prox_fs, prox_g, k_exp, y)
    assert_array_almost_equal(y, x_rec)

    # observations of size (5,2)
    y = np.zeros((5, 2))
    x_rec = admm(prox_fs, prox_g, k_nd, y)
    assert_array_almost_equal(y, x_rec)
    x_rec = admm(prox_fs, prox_g, k_exp, y)
    assert_array_almost_equal(y, x_rec)


def test_fb_virtual_zero():
    # Virtual 0-prox
    prox_f = lambda u, la: 0 * 0
    grad_g = lambda u: u * 0

    # observations of size (5,1)
    y = np.zeros((5, 1))
    for method in methods:
        x_rec = forward_backward(prox_f, grad_g, y, 1, method=method)
        assert_array_almost_equal(y, x_rec)

    # observations of size (5,2)
    y = np.zeros((5, 2))
    for method in methods:
        x_rec = forward_backward(prox_f, grad_g, y, 1, method=method)
        assert_array_almost_equal(y, x_rec)


def test_fb_zero():
    prox_f = lambda u, la: u
    grad_g = lambda u: u * 0

    # observations of size (5,1)
    y = np.zeros((5, 1))
    for method in methods:
        x_rec = forward_backward(prox_f, grad_g, y, 1, method=method)
        assert_array_almost_equal(y, x_rec)

    # observations of size (5,2)
    y = np.zeros((5, 2))
    for method in methods:
        x_rec = forward_backward(prox_f, grad_g, y, 1, method=method)
        assert_array_almost_equal(y, x_rec)


def test_fb_l1_denoising():
    n = 1000
    # Use a very sparse vector for the test
    x = np.zeros((n, 1))
    x[1, :] = 100
    y = x + 0.06 * np.random.randn(n, 1)

    la = 0.2
    prox_f = lambda x, tau: soft_thresholding(x, la * tau)
    grad_g = lambda x: x - y

    for method in methods:
        x_rec = forward_backward(prox_f, grad_g, y, 1, method=method)
        #TODO ugly test to change
        assert_array_almost_equal(x, x_rec, decimal=0)


def test_fb_dual_virtual_zero():
    # Virtual 0-prox
    grad_fs = lambda u: u * 0
    prox_gs = lambda u, la: u * 0

    # ndarray
    k_nd = np.zeros((5, 5))
    # explicit
    k_exp = lambda u: 0 * u
    k_exp.T = lambda u: 0 * u

    # observations of size (5,1)
    y = np.zeros((5, 1))
    x_rec = forward_backward_dual(grad_fs, prox_gs, k_nd, y, 1)
    assert_array_almost_equal(y, x_rec)
    x_rec = forward_backward_dual(grad_fs, prox_gs, k_exp, y, 1)
    assert_array_almost_equal(y, x_rec)

    # observations of size (5,2)
    y = np.zeros((5, 2))
    x_rec = forward_backward_dual(grad_fs, prox_gs, k_nd, y, 1)
    assert_array_almost_equal(y, x_rec)
    x_rec = forward_backward_dual(grad_fs, prox_gs, k_exp, y, 1)
    assert_array_almost_equal(y, x_rec)
