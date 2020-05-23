import numpy as np
import os
import sys
sys.path.append(os.path.abspath(r'../lib'))
import NumCpp  # noqa E402


####################################################################################
def test_constants():
    """Tests the NumCpp constants"""
    NUM_DECIMALS_ROUND = 10

    assert round(NumCpp.c, NUM_DECIMALS_ROUND) == round(3e8, 10)
    assert round(NumCpp.e, NUM_DECIMALS_ROUND) == round(np.e, 10)
    assert np.isinf(NumCpp.inf)
    assert round(NumCpp.pi, NUM_DECIMALS_ROUND) == round(np.pi, 10)
    assert np.isnan(NumCpp.nan)
    assert NumCpp.j == 1j
    assert NumCpp.VERSION == '1.4.0'
