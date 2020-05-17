import numpy as np
import os
import sys
sys.path.append(os.path.abspath(r'../lib'))
import NumCpp  # noqa E402


####################################################################################
def test_DtypeInfo():
    """Tests the NumCpp DtypeInfo"""
    assert NumCpp.DtypeIntoUint32.bits() == 32
    assert NumCpp.DtypeIntoUint32.epsilon() == 0
    assert NumCpp.DtypeIntoUint32.isInteger()
    assert NumCpp.DtypeIntoUint32.max() == np.iinfo(np.uint32).max
    assert NumCpp.DtypeIntoUint32.min() == np.iinfo(np.uint32).min
