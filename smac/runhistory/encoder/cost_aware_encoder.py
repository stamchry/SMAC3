from __future__ import annotations

from smac.runhistory.encoder.eips_encoder import RunHistoryEIPSEncoder

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class RunHistoryCostAwareEncoder(RunHistoryEIPSEncoder):
    """
    Encoder for the CostAwareModel. It builds a data matrix `Y` with two columns:
    the first for the aggregated performance objective(s) and the second for the cost objective.
    This class is a child of the RunHistoryEIPSEncoder and is used for cost-aware optimization.
    """

    pass
