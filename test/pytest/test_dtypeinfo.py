import numpy as np
import os
import sys

import NumCppPy as NumCpp  # noqa E402


####################################################################################
def test_uint32_bits():
    assert NumCpp.DtypeIntoUint32.bits() == 32


####################################################################################
def test_uint32_epsilon():
    assert NumCpp.DtypeIntoUint32.epsilon() == 0


####################################################################################
def test_uint32_isInteger():
    assert NumCpp.DtypeIntoUint32.isInteger()


####################################################################################
def test_uint32_isSigned():
    assert not NumCpp.DtypeIntoUint32.isSigned()


####################################################################################
def test_uint32_max():
    assert NumCpp.DtypeIntoUint32.max() == np.iinfo(np.uint32).max


####################################################################################
def test_uint32_min():
    assert NumCpp.DtypeIntoUint32.min() == np.iinfo(np.uint32).min


####################################################################################
def test_complex_bits():
    assert NumCpp.DtypeInfoComplexDouble.bits()


####################################################################################
def test_complex_epsilon():
    assert NumCpp.DtypeInfoComplexDouble.epsilon()


####################################################################################
def test_complex_isInteger():
    assert not NumCpp.DtypeInfoComplexDouble.isInteger()


####################################################################################
def test_complex_isSigned():
    assert NumCpp.DtypeInfoComplexDouble.isSigned()


####################################################################################
def test_complex_max():
    assert NumCpp.DtypeInfoComplexDouble.max()


####################################################################################
def test_complex_min():
    assert NumCpp.DtypeInfoComplexDouble.min()
