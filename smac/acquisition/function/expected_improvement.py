from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.model.cost_aware_model import CostAwareModel
from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class EI(AbstractAcquisitionFunction):
    r"""The Expected Improvement (EI) criterion is used to decide where to evaluate a function f(x) next. The goal is to
    balance exploration and exploitation. Expected Improvement (with or without function values in log space)
    acquisition function

    :math:`EI(X) := \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi \} \right]`,
    with :math:`f(X^+)` as the best location.

    Reference for EI: Jones, D.R. and Schonlau, M. and Welch, W.J. (1998). Efficient Global Optimization of Expensive
    Black-Box Functions. Journal of Global Optimization 13, 455–492

    Reference for logEI: Hutter, F. and Hoos, H. and Leyton-Brown, K. and Murphy, K. (2009). An experimental
    investigation of model-based parameter optimisation: SPO and beyond. In: Conference on Genetic and
    Evolutionary Computation

    The logEI implemententation is based on the derivation of the orginal equation by:
    Watanabe, S. (2024). Derivation of Closed Form of Expected Improvement for Gaussian Process Trained on
    Log-Transformed Objective. https://arxiv.org/abs/2411.18095

    Parameters
    ----------
    xi : float, defaults to 0.0
        Controls the balance between exploration and exploitation of the
        acquisition function.
    log : bool, defaults to False
        Whether the function values are in log-space.


    Attributes
    ----------
    _xi : float
        Exploration-exloitation trade-off parameter.
    _log: bool
        Function values in log-space or not.
    _eta : float
        Current incumbent function value (best value observed so far).

    """

    def __init__(
        self,
        xi: float = 0.0,
        log: bool = False,
    ) -> None:
        super(EI, self).__init__()

        self._xi: float = xi
        self._log: bool = log
        self._eta: float | None = None

    @property
    def name(self) -> str:  # noqa: D102
        return "Expected Improvement"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "xi": self._xi,
                "log": self._log,
            }
        )

        return meta

    def _update(self, **kwargs: Any) -> None:
        """Update acsquisition function attributes

        Parameters
        ----------
        eta : float
            Function value of current incumbent.
        xi : float, optional
            Exploration-exploitation trade-off parameter
        """
        assert "eta" in kwargs
        self._eta = kwargs["eta"]

        if "xi" in kwargs and kwargs["xi"] is not None:
            self._xi = kwargs["xi"]

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Compute EI acquisition value

        Parameters
        ----------
        X : np.ndarray [N, D]
            The input points where the acquisition function should be evaluated. The dimensionality of X is (N, D),
            with N as the number of points to evaluate at and D is the number of dimensions of one X.

        Returns
        -------
        np.ndarray [N,1]
            Acquisition function values wrt X.

        Raises
        ------
        ValueError
            If `update` has not been called before (current incumbent value `eta` unspecified).
        ValueError
            If EI is < 0 for at least one sample (normal function value space).
        ValueError
            If EI is < 0 for at least one sample (log function value space).
        """
        assert self._model is not None
        assert self._xi is not None

        if self._eta is None:
            raise ValueError(
                "No current best specified. Call update("
                "eta=<int>) to inform the acquisition function "
                "about the current best value."
            )

        if not self._log:
            if len(X.shape) == 1:
                X = X[:, np.newaxis]

            m, v = self._model.predict_marginalized(X)
            s = np.sqrt(v)

            def calculate_f() -> np.ndarray:
                z = (self._eta - m - self._xi) / s
                return (self._eta - m - self._xi) * norm.cdf(z) + s * norm.pdf(z)

            if np.any(s == 0.0):
                # if std is zero, we have observed x on all instances
                # using a RF, std should be never exactly 0.0
                # Avoid zero division by setting all zeros in s to one.
                # Consider the corresponding results in f to be zero.
                logger.warning("Predicted std is 0.0 for at least one sample.")
                s_copy = np.copy(s)
                s[s_copy == 0.0] = 1.0
                f = calculate_f()
                f[s_copy == 0.0] = 0.0
            else:
                f = calculate_f()

            if (f < 0).any():
                raise ValueError("Expected Improvement is smaller than 0 for at least one " "sample.")

            return f
        else:
            if len(X.shape) == 1:
                X = X[:, np.newaxis]

            m, var_ = self._model.predict_marginalized(X)
            std = np.sqrt(var_)

            def calculate_log_ei() -> np.ndarray:
                # we expect that f_min is in log-space
                assert self._eta is not None
                assert self._xi is not None

                f_min = self._eta - self._xi
                v = (f_min - m) / std
                return (np.exp(f_min) * norm.cdf(v)) - (np.exp(0.5 * var_ + m) * norm.cdf(v - std))

            if np.any(std == 0.0):
                # if std is zero, we have observed x on all instances
                # using a RF, std should be never exactly 0.0
                # Avoid zero division by setting all zeros in s to one.
                # Consider the corresponding results in f to be zero.
                logger.warning("Predicted std is 0.0 for at least one sample.")
                std_copy = np.copy(std)
                std[std_copy == 0.0] = 1.0
                log_ei = calculate_log_ei()
                log_ei[std_copy == 0.0] = 0.0
            else:
                log_ei = calculate_log_ei()

            if (log_ei < 0).any():
                raise ValueError("Expected Improvement is smaller than 0 for at least one sample.")

            return log_ei.reshape((-1, 1))


class EICool(EI):
    """EI-cool acquisition function.

    This acquisition function uses a cooling schedule for the cost exponent `alpha`.
    When the remaining budget is high, `alpha` is close to 0, prioritizing improvement.
    As the budget depletes, `alpha` approaches 1, prioritizing cheaper configurations.
    If `alpha` is set to a fixed value, the cooling schedule is disabled.

    Parameters
    ----------
    scenario: Scenario
        The scenario object, which contains the budget information.
    xi : float, defaults to 0.0
        Controls the balance between exploration and exploitation.
    log : bool, defaults to False
        Whether the function values are in log-space.
    alpha : float | None, defaults to None
        The exponent for the cost. If None, a cooling schedule is used.
        If a float, `alpha` is fixed to that value.
    """

    def __init__(
        self,
        scenario: Scenario,
        xi: float = 0.0,
        log: bool = False,
        alpha: float | None = None,
    ):
        super().__init__(xi=xi, log=log)
        self._scenario = scenario
        self._consumed_budget: float = 0.0
        self._fixed_alpha = alpha

        if self._fixed_alpha is None:
            if not self._scenario.cost_aware or self._scenario.cost_aware_budget is None:
                raise ValueError("EICool with cooling schedule requires a cost_aware_budget to be set.")

            # We get the initial budget from the switching acquisition function's default
            self._tau = self._scenario.cost_aware_budget
            self._tau_init = self._scenario.cost_aware_budget * (1 / 8)

    @property
    def name(self) -> str:  # noqa: D102
        return "Expected Improvement Cost Cooling"

    def _update(self, **kwargs: Any) -> None:
        """Updates the consumed budget."""
        super()._update(**kwargs)
        if not isinstance(self._model, CostAwareModel):
            raise TypeError("EICool requires a CostAwareModel.")

        if "consumed_budget" in kwargs:
            self._consumed_budget = kwargs["consumed_budget"]

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Compute the EI-cool value."""
        if not isinstance(self._model, CostAwareModel):
            raise TypeError("EICool requires a CostAwareModel.")

        ei_values = super()._compute(X)

        cost_values, _ = self._model.predict_cost(X)
        cost_values = np.maximum(cost_values, 1e-9)  # Avoid division by zero

        if self._fixed_alpha is not None:
            alpha = self._fixed_alpha
        else:
            tau_k = self._consumed_budget
            if self._tau <= self._tau_init:
                alpha = 1.0
            else:
                alpha = (self._tau - tau_k) / (self._tau - self._tau_init)

            # Alpha should be between 0 and 1
            alpha = np.clip(alpha, 0, 1)

        ei_cool_values = ei_values / (cost_values**alpha)

        return ei_cool_values


class EIPS(EICool):
    r"""Expected Improvement per Second acquisition function.

    This is equivalent to EICool with a fixed alpha of 1.0.

    :math:`EI(X) := \frac{\mathbb{E}\left[\max\{0,f(\mathbf{X^+})-f_{t+1}(\mathbf{X})-\xi\right]\}]}{c(x)}`,
    with :math:`f(X^+)` as the best location and :math:`c(x)` as the cost.

    Parameters
    ----------
    scenario: Scenario
        The scenario object.
    xi : float, defaults to 0.0
        Controls the balance between exploration and exploitation.
    log : bool, defaults to False
        Whether the function values are in log-space.
    """

    def __init__(self, scenario: Scenario, xi: float = 0.0, log: bool = False):
        super().__init__(scenario=scenario, xi=xi, log=log, alpha=1.0)

    @property
    def name(self) -> str:  # noqa: D102
        return "Expected Improvement per Second"
