import numpy as np
import pandas as pd
import pytest

import hydrosignatures as hs
from hydrosignatures import InputTypeError


def test_same_input_length():
    q_mmpt, p_mmpt = pd.Series(np.arange(10)), pd.Series(np.arange(11))
    with pytest.raises(InputTypeError, match="same length"):
        _ = hs.HydroSignatures(q_mmpt, p_mmpt)
