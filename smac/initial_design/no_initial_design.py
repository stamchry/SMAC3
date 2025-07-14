from __future__ import annotations

from ConfigSpace import Configuration

from smac.initial_design.abstract_initial_design import AbstractInitialDesign


class NoInitialDesign(AbstractInitialDesign):
    """This initial design does not return any configurations. It is used when the
    initial design is handled by another component, e.g., a switching acquisition function.
    """

    def _select_configurations(self) -> list[Configuration]:
        """Returns an empty list of configurations."""
        return []
