"""Quote-cleaning contracts for the surface input layer."""

import numpy as np
import pandas as pd
import pytest

from src.data import clean_chain


def _quotes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # valid quote
            {"bid": 1.0, "ask": 1.1, "mid": 1.05, "open_interest": 20, "T": 30 / 365},
            # one row per quality failure
            {"bid": 0.05, "ask": 0.10, "mid": 0.075, "open_interest": 20, "T": 30 / 365},
            {"bid": 1.0, "ask": 1.5, "mid": 1.25, "open_interest": 20, "T": 30 / 365},
            {"bid": 1.0, "ask": 1.1, "mid": 1.05, "open_interest": 9, "T": 30 / 365},
            {"bid": 1.0, "ask": 1.1, "mid": 1.05, "open_interest": 20, "T": 4 / 365},
            {"bid": 1.0, "ask": 1.1, "mid": np.nan, "open_interest": 20, "T": 30 / 365},
        ]
    )


def test_clean_chain_keeps_only_finite_liquid_quotes():
    cleaned = clean_chain(_quotes())
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["mid"] == 1.05


def test_clean_chain_requires_surface_input_columns():
    with pytest.raises(ValueError, match="open_interest"):
        clean_chain(pd.DataFrame({"bid": [1.0], "ask": [1.1], "mid": [1.05], "T": [0.1]}))
