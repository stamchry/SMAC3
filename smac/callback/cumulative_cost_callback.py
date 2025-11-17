from __future__ import annotations

from smac.callback.callback import Callback
from smac.main.smbo import SMBO
from smac.runhistory.dataclasses import TrialInfo, TrialValue

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class CumulativeCostCallback(Callback):
    """Callback to track the cumulative cost of all evaluated trials."""

    def __init__(
        self,
        cumulative_cost_tracker: list[float],
    ):
        self._cumulative_cost_tracker = cumulative_cost_tracker

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        """
        Called after the trial is finished and the runhistory is updated.
        We use the trial's cost to update our cumulative cost tracker.
        """
        # The cost is stored in the additional info of the trial value
        cost = value.additional_info.get("resource_cost", 0.0)
        self._cumulative_cost_tracker[0] += cost

        return None
