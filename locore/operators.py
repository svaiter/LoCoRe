from __future__ import division
import numpy as np


# Thresholding
def soft_thresholding(x, gamma):
    return np.maximum(0, 1 - gamma / np.maximum(np.abs(x), 1E-10)) * x


def dual_prox(prox):
    return lambda u, sigma: u - sigma * prox(u / sigma, 1 / sigma)


# Group related
def linf_l2_norm(blocks, x, eps=1e-6):
    res = 0.0
    tmp = 0.0
    for i in range(blocks.shape[0]):
        tmp = np.linalg.norm(x[blocks[i, :]])
        if tmp > res:
            res, tmp = tmp, res
    return res


def l1_l2_normalize_subvector(blocks, support, x):
    e = np.zeros((sum(support), x.shape[1]))
    for i in range(blocks.shape[0]):
        if support[blocks[i, :]][0] is True:
            e[blocks[i, :]] = x[blocks[i, :]] / np.linalg.norm(x[blocks[i, :]])
    return e


# Support
def l1_support(x, eps=1e-6):
    return (np.abs(x) > eps).flatten()


def linf_support(x, eps=1e-6):
    max_value = np.max(np.abs(x))
    return (np.abs(np.abs(x) - max_value) < eps).flatten()


def l1_l2_support(blocks, x, eps=1e-6):
    support = np.zeros(x.shape, dtype=bool)
    for i in range(blocks.shape[0]):
        if np.linalg.norm(x[blocks[i, :]]) > eps:
            support[blocks[i, :]] = True
    return support.flatten()


# Dictionaries
def finite_diff_1d(n, bound='sym'):
    D = np.eye(n) - np.diag(np.ones((n-1,)), 1)
    if bound == 'sym':
        D = D[:, 1:]
    elif bound == 'per':
        pass
    else:
        raise Exception('Not a valid boundary condition')
    return D


def fused_lasso_dict(n, eps=0.5):
    return np.concatenate((finite_diff_1d(n), eps * np.eye(n)), 1)
