from __future__ import division
import numpy as np

from .solvers import (
    solve_l1_synthesis_lagrangian,
    solve_l1_synthesis_constrained,
    solve_l1_analysis
)
from .identifiability import crit_l1_synthesis, crit_l1_analysis
from .operators import l1_support


class Sparse(object):

    def __init__(self, Phi):
        self.Phi = Phi

    def tangent_model(self, x):
        support = l1_support(x)
        return np.eye(x.shape[0])[:, support]

    def solve_l2(self, la, y):
        return solve_l1_synthesis_lagrangian(self.Phi, y, la=la, maxiter=500)

    def solve_noiseless(self, y):
        return solve_l1_synthesis_constrained(self.Phi, y, maxiter=500)

    def ic(self, x):
        return crit_l1_synthesis(self.Phi, x)[0]

    def linearized_precertificate(self, x):
        return crit_l1_synthesis(self.Phi, x)[1]


class AnalysisSparse(object):

    def __init__(self, DS, Phi):
        self.DS = DS
        self.Phi = Phi

    def tangent_model(self, x):
        corr = np.dot(self.DS, x)
        support = l1_support(corr)
        return np.eye(corr.shape[0])[:, support]

    def solve_l2(self, la, y):
        return solve_l1_analysis(self.DS, self.Phi, y, la=la, maxiter=500, constrained=False)

    def solve_noiseless(self, y):
        return solve_l1_analysis(self.DS, self.Phi, y, la=0.0, maxiter=500, constrained=True)

    def ic(self, x):
        return crit_l1_analysis(self.DS, self.Phi, x)[0]

    def linearized_precertificate(self, x):
        return crit_l1_analysis(self.DS, self.Phi, x)[1]
