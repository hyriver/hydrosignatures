import numpy as np
import pandas as pd
import pytest

import hydrosignatures as hs
from hydrosignatures import InputTypeError


def test_same_input_length():
    with pytest.raises(InputTypeError) as ex:
        _ = hs.HydroSignatures(pd.Series(np.arange(10)), pd.Series(np.arange(11)))
        assert "same length" in str(ex.value)
