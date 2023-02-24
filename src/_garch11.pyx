import numpy as np
from xalglib import xalglib
from sklearn.base import BaseEstimator
cimport numpy as np

from _garch11 cimport calc_fun_jac as c_calc_fun_jac
from _garch11 cimport calc_fun_jac_bc as c_calc_fun_jac_bc
from _garch11 cimport transform as c_transform
from _garch11 cimport transform_bc as c_transform_bc
from _garch11 cimport predict as c_predict
from _garch11 cimport nll as c_nll


def calc_fun_jac(np.ndarray[np.double_t] x, np.ndarray[np.double_t] r2):
    cdef int n = r2.shape[0]
    x = np.ascontiguousarray(x)
    r2 = np.ascontiguousarray(r2)
    cdef np.ndarray[np.double_t] out_jac = np.empty_like(x)
    out_jac = np.ascontiguousarray(out_jac)
    cdef double y = c_calc_fun_jac(&x[0], &r2[0], n, &out_jac[0])
    return y, out_jac


def nsfunc_jac(x, fi, jac, param):
    cdef np.ndarray[np.double_t] nx = np.asarray(x)
    cdef np.ndarray[np.double_t] nr2 = np.asarray(param)
    y, out_jac = calc_fun_jac(nx, nr2)
    fi[0] = y
    jac[0][0] = out_jac[0]
    jac[0][1] = out_jac[1]
    jac[0][2] = out_jac[2]


def calc_fun_jac_bc(np.ndarray[np.double_t] x, np.ndarray[np.double_t] r2):
    cdef int n = r2.shape[0]
    x = np.ascontiguousarray(x)
    r2 = np.ascontiguousarray(r2)
    cdef np.ndarray[np.double_t] out_jac = np.empty_like(x)
    out_jac = np.ascontiguousarray(out_jac)
    cdef double y = c_calc_fun_jac_bc(&x[0], &r2[0], n, &out_jac[0])
    return y, out_jac


def nsfunc_jac_bc(x, fi, jac, param):
    cdef np.ndarray[np.double_t] nx = np.asarray(x)
    cdef np.ndarray[np.double_t] nr2 = np.asarray(param)
    y, out_jac = calc_fun_jac_bc(nx, nr2)
    fi[0] = y
    jac[0][0] = out_jac[0]
    jac[0][1] = out_jac[1]
    jac[0][2] = out_jac[2]


def transform(np.ndarray[np.double_t] x, np.ndarray[np.double_t] r2):
    cdef int n = r2.shape[0]
    x = np.ascontiguousarray(x)
    r2 = np.ascontiguousarray(r2)
    cdef np.ndarray[np.double_t] out_sigma2 = np.empty_like(r2)
    out_sigma2 = np.ascontiguousarray(out_sigma2)
    c_transform(&x[0], &r2[0], n, &out_sigma2[0])
    return out_sigma2


def transform_bc(np.ndarray[np.double_t] x, np.ndarray[np.double_t] r2):
    cdef int n = r2.shape[0]
    x = np.ascontiguousarray(x)
    r2 = np.ascontiguousarray(r2)
    cdef np.ndarray[np.double_t] out_sigma2 = np.empty_like(r2)
    out_sigma2 = np.ascontiguousarray(out_sigma2)
    c_transform_bc(&x[0], &r2[0], n, &out_sigma2[0])
    return out_sigma2


def predict(np.ndarray[np.double_t] x, double last_sigma2, double last_r2,
            np.ndarray[np.double_t] randn_nums):
    cdef int n = randn_nums.shape[0]
    x = np.ascontiguousarray(x)
    cdef np.ndarray[np.double_t] out_sigma2 = np.empty_like(randn_nums)
    out_sigma2 = np.ascontiguousarray(out_sigma2)
    cdef np.ndarray[np.double_t] out_r2 = np.empty_like(randn_nums)
    out_r2 = np.ascontiguousarray(out_r2)
    c_predict(&x[0], last_sigma2, last_r2, &randn_nums[0], n, &out_sigma2[0],
              &out_r2[0])
    return out_sigma2, out_r2


def nll(np.ndarray[np.double_t] r2, np.ndarray[np.double_t] sigma2):
    cdef int n = r2.shape[0]
    r2 = np.ascontiguousarray(r2)
    sigma2 = np.ascontiguousarray(sigma2)
    return c_nll(&r2[0], n, &sigma2[0])


# Don't check this class against
# sklearn.utils.estimator_checks.check_estimator!
# It doesn't satisfy the convention.
class Garch11(BaseEstimator):
    """
    Parameters (set at __init__ or via set_params):

        - backcast: True to use backcast initialization; False to use mean
                    initialization

    Attributes (readonly):

        - omega_: the optimization result, omega
        - alpha_: the optimization result, alpha
        - beta_: the optimization result, beta
        - backcast_: whether the optimization is using backcast initialization
        - rep_: previous optimization report
    """
    @staticmethod
    def _get_x0():
        # According to https://math.berkeley.edu/~btw/thesis4.pdf,
        # omega > 0, alpha > 1, 0 < beta < 1, alpha != beta, alpha + beta < 1
        # omega is mostly small
        return [1e-6, 0.4, 0.5]

    def __init__(self, backcast=False):
        self.backcast = bool(backcast)

    def get_params(self, deep=True):
        return {
            'backcast': self.backcast,
        }

    def set_params(self, **params):
        if 'backcast' in params:
            self.backcast = bool(params['backcast'])
        return self

    def __repr__(self):
        return f'Garch11(backcast={self.backcast})'

    def fit(self, r2: np.ndarray):
        """
        :param r2: square of log returns
        """
        assert r2.shape[0] > 1 if self.backcast else r2.shape[0] > 0, \
               'too few elements in r2: ' + str(r2.shape[0])
        if self.backcast:
            x, rep = self._fit_AGS_backcast(r2)
        else:
            x, rep = self._fit_AGS(r2)
        self.x = np.asarray(x)
        self.rep_ = rep
        return self

    def _fit_AGS(self, r2):
        state = xalglib.minnscreate(3, Garch11._get_x0())
        xalglib.minnssetcond(state, 1e-6, 0)
        xalglib.minnssetscale(state, [1, 1, 1])
        xalglib.minnssetalgoags(state, 0.1, 0.0)  # no nonlinear constraint
        # lower bounds and upper bounds
        xalglib.minnssetbc(state, [1e-9, 1e-6, 1e-6], [5, 1, 1])
        xalglib.minnssetlc(state, [[0, 1, 1, 0.999999]], [-1], 1)
        xalglib.minnsoptimize_j(state, nsfunc_jac, param=r2)
        return xalglib.minnsresults(state)

    def _fit_AGS_backcast(self, r2):
        state = xalglib.minnscreate(3, Garch11._get_x0())
        xalglib.minnssetcond(state, 1e-6, 0)
        xalglib.minnssetscale(state, [1, 1, 1])
        xalglib.minnssetalgoags(state, 0.1, 0.0)  # no nonlinear constraint
        # lower bounds and upper bounds
        xalglib.minnssetbc(state, [1e-9, 1e-6, 1e-6], [5, 1, 1])
        xalglib.minnssetlc(state, [[0, 1, 1, 0.999999]], [-1], 1)
        xalglib.minnsoptimize_j(state, nsfunc_jac_bc, param=r2)
        return xalglib.minnsresults(state)

    def transform(self, r2: np.ndarray):
        """
        :param r2: square of log returns
        :return: sigma2
        """
        if self.backcast:
            sigma2 = transform_bc(self.x, r2)
        else:
            sigma2 = transform(self.x, r2)
        return sigma2

    def fit_transform(self, r2: np.ndarray):
        return self.fit(r2).transform(r2)

    def predict(self, last_sigma2: float, last_r2: float,
                randn_nums: np.ndarray, return_r2=False):
        """
        :param last_sigma2: last sigma2
        :param last r2: last log return
        :param randn_nums: an array of standard normal random variables of
               length n
        :param return_r2: True to return (future_sigma2, future_r2); False
               (the default) to return future_sigma2
        :return: future_sigma2 (and future_r2 if return_r2 is True) of length
                 n
        """
        out_sigma2, out_r2 = predict(self.x, last_sigma2, last_r2, randn_nums)
        if return_r2:
            return out_sigma2, out_r2
        return out_sigma2

    def nll(self, r2: np.ndarray, sigma2: np.ndarray):
        """
        Negative log likelihood.
        :param r2: square of log returns
        :param sigma2: the transform result
        """
        return nll(r2, sigma2)

    def fit_transform_nll(self, r2):
        sigma2 = self.fit_transform(r2)
        return self.nll(r2, sigma2)

    @property
    def omega_(self):
        return self.x[0]

    @property
    def alpha_(self):
        return self.x[1]

    @property
    def beta_(self):
        return self.x[2]

    @property
    def backcast_(self):
        return self.backcast
