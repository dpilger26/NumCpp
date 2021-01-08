import numpy as np
import os
import re
import sys
sys.path.append(os.path.abspath(r'../lib'))
import NumCppPy as NumCpp  # noqa E402

NUM_DECIMALS_ROUND = 10


####################################################################################
def test_version():
    version_file = r'../../include/NumCpp/Core/Internal/Version.hpp'
    with open(version_file, 'r') as fid:
        version_contents = fid.read()

    version_re = re.compile('constexpr[ ]char[ ]VERSION\\[\\][ ]=[ ]\"(.+)\"')  # noqa
    version_toks = version_re.findall(version_contents)
    assert len(version_toks) == 1, 'Unable to parse the version number'
    assert NumCpp.VERSION == version_toks[0]


####################################################################################
def test_c():

    assert round(NumCpp.c, NUM_DECIMALS_ROUND) == round(3e8, 10)


####################################################################################
def test_e():
    assert round(NumCpp.e, NUM_DECIMALS_ROUND) == round(np.e, 10)


####################################################################################
def test_constants():
    assert np.isinf(NumCpp.inf)


####################################################################################
def test_pi():
    assert round(NumCpp.pi, NUM_DECIMALS_ROUND) == round(np.pi, 10)


####################################################################################
def test_nan():
    assert np.isnan(NumCpp.nan)


####################################################################################
def test_j():
    assert NumCpp.j == 1j


####################################################################################
def test_DAYS_PER_WEEK():
    assert NumCpp.DAYS_PER_WEEK == 7


####################################################################################
def test_MINUTES_PER_HOUR():
    assert NumCpp.MINUTES_PER_HOUR == 60


####################################################################################
def test_SECONDS_PER_MINUTE():
    assert NumCpp.SECONDS_PER_MINUTE == 60


####################################################################################
def test_MILLISECONDS_PER_SECOND():
    assert NumCpp.MILLISECONDS_PER_SECOND == 1000


####################################################################################
def test_SECONDS_PER_HOUR():
    assert NumCpp.SECONDS_PER_HOUR == 3600


####################################################################################
def test_HOURS_PER_DAY():
    assert NumCpp.HOURS_PER_DAY == 24


####################################################################################
def test_MINUTES_PER_DAY():
    assert NumCpp.MINUTES_PER_DAY == 1440


####################################################################################
def test_SECONDS_PER_DAY():
    assert NumCpp.SECONDS_PER_DAY == 86400


####################################################################################
def test_MILLISECONDS_PER_DAY():
    assert NumCpp.MILLISECONDS_PER_DAY == 86400000


####################################################################################
def test_SECONDS_PER_WEEK():
    assert NumCpp.SECONDS_PER_WEEK == 604800
