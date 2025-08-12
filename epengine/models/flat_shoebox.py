"""A model for a flat parameter shoebox simulation."""

import logging
from collections.abc import Callable

import pandas as pd
from epinterface.sbem.flat_model import FlatModel

from epengine.models.base import LeafSpec

logger = logging.getLogger(__name__)


class FlatShoeboxSimulationSpec(LeafSpec, FlatModel):
    """A spec for running a flat shoebox simulation."""

    def run(self, log_fn: Callable | None = None) -> pd.DataFrame:
        """Run the simulation."""
        log_fn = log_fn or logger.info
        log_fn("Running flat shoebox simulation...")
        _idf, results, _err_text = self.simulate()
        log_fn("Flat shoebox simulation complete.")

        dumped_self = self.model_dump()
        index = pd.MultiIndex.from_tuples(
            [tuple(dumped_self.values())],
            names=list(dumped_self.keys()),
        )
        results = results.to_frame().T.set_index(index)
        return results
