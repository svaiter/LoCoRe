from __future__ import division
import numpy as np

from .solvers import (
    solve_l1_synthesis_lagrangian,
    solve_l1_synthesis_constrained,
    solve_l1_analysis
)
from .identifiability import (
    crit_l1_synthesis,
    crit_l1_analysis,
    crit_nuclear,
    crit_linf
)
from .operators import l1_support


class Sparse(object):
    """
    Sparse models through l^1 minimization.

    Parameters
    ----------
    Phi : ndarray
        forward operator.
    """

    def __init__(self, Phi):
        self.Phi = Phi

    def tangent_model(self, x):
        """Return an orthogonal basis of the model tangent space for sparsity.

        Parameters
        ----------
        x : ndarray
            sparse vector.

        Notes
        -----
        This basis is computed from the support of the vector x.
        """
        # TODO: orthogonalize the basis
        support = l1_support(x)
        return np.eye(x.shape[0])[:, support]

    def solve_l2(self, la, y):
        """Compute the solution of the Lasso.

        Parameters
        ----------
        la : float
            hyper-parameter of the Lasso.
        y : ndarray
            observations.

        Notes
        -----
        This solution is computed with a forward-backward algorithm.
        """
        return solve_l1_synthesis_lagrangian(self.Phi, y, la=la, maxiter=500)

    def solve_noiseless(self, y):
        """Compute the solution of the Basis Pursuit.

        Parameters
        ----------
        y : ndarray
            observations.

        Notes
        -----
        This solution is computed with a Douglas-Rachford algorithm.
        """
        return solve_l1_synthesis_constrained(self.Phi, y, maxiter=500)

    def ic(self, x):
        """Compute the Identifiability Criterion of a sparse vector.

        Parameters
        ----------
        x : ndarray
            sparse vector.
        """
        return crit_l1_synthesis(self.Phi, x)[0]

    def linearized_precertificate(self, x):
        """Compute the linearized precertificate of a sparse vector.

        Parameters
        ----------
        x : ndarray
            sparse vector.
        """
        return crit_l1_synthesis(self.Phi, x)[1]


class AnalysisSparse(object):
    """
    Analysis sparse models through generalized l^1 minimization.

    Parameters
    ----------
    DS : ndarray
        dictionary.
    Phi : ndarray
        forward operator.
    """

    def __init__(self, DS, Phi):
        self.DS = DS
        self.Phi = Phi

    def tangent_model(self, x):
        """Return an orthogonal basis of the model tangent space for sparsity.

        Parameters
        ----------
        x : ndarray
            sparse vector.

        Notes
        -----
        This basis is computed from the support of the vector DS x.
        """
        corr = np.dot(self.DS, x)
        support = l1_support(corr)
        return np.eye(corr.shape[0])[:, support]

    def solve_l2(self, la, y):
        """Compute the solution of the Analysis Lasso.

        Parameters
        ----------
        la : float
            hyper-parameter of the Analysis Lasso.
        y : ndarray
            observations.

        Notes
        -----
        This solution is computed with an ADMM algorithm.
        """
        return solve_l1_analysis(self.DS, self.Phi, y, la=la,
                                 maxiter=500, constrained=False)

    def solve_noiseless(self, y):
        """Compute the solution of the Analysis Basis Pursuit.

        Parameters
        ----------
        y : ndarray
            observations.

        Notes
        -----
        This solution is computed with an ADMM algorithm.
        """
        return solve_l1_analysis(self.DS, self.Phi, y, la=0.0,
                                 maxiter=500, constrained=True)

    def ic(self, x):
        """Compute the Identifiability Criterion of a sparse vector.

        Parameters
        ----------
        x : ndarray
            sparse vector.
        """
        return crit_l1_analysis(self.DS, self.Phi, x)[0]

    def linearized_precertificate(self, x):
        """Compute the linearized precertificate of a sparse vector.

        Parameters
        ----------
        x : ndarray
            sparse vector.
        """
        return crit_l1_analysis(self.DS, self.Phi, x)[1]


class LowRank(object):
    """
    Low rank models through nuclear norm minimization.

    Parameters
    ----------
    Phi : ndarray
        forward operator.
    """

    def __init__(self, Phi):
        self.Phi = Phi

    def tangent_model(self, x):
        return None

    def solve_l2(self, la, y):
        return None

    def solve_noiseless(self, y):
        return None

    def ic(self, x):
        return crit_nuclear(self.Phi, x)[0]

    def linearized_precertificate(self, x):
        return crit_nuclear(self.Phi, x)[1]


class AntiSparse(object):
    """
    Antisparse/flat models through l^inf minimization.

    Parameters
    ----------
    Phi : ndarray
        forward operator.
    """

    def __init__(self, Phi):
        self.Phi = Phi

    def tangent_model(self, x):
        return None

    def solve_l2(self, la, y):
        return None

    def solve_noiseless(self, y):
        return None

    def ic(self, x):
        return crit_linf(self.Phi, x)[0]

    def linearized_precertificate(self, x):
        return crit_linf(self.Phi, x)[1]
