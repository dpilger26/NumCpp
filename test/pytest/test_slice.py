import numpy as np
import os
import sys
sys.path.append(os.path.abspath(r'../lib'))
import NumCpp  # noqa E402


####################################################################################
def test_slice():
    """Tests the NumCpp Slice class"""
    np.random.seed(666)

    cSlice = NumCpp.Slice()
    assert cSlice.start == 0
    assert cSlice.stop == 1
    assert cSlice.step == 1

    stop = np.random.randint(0, 100, [1, ]).item()
    cSlice = NumCpp.Slice(stop)
    assert cSlice.start == 0
    assert cSlice.stop == stop
    assert cSlice.step == 1

    start = np.random.randint(0, 100, [1, ]).item()
    stop = np.random.randint(100, 200, [1, ]).item()
    cSlice = NumCpp.Slice(start, stop)
    assert cSlice.start == start
    assert cSlice.stop == stop
    assert cSlice.step == 1

    start = np.random.randint(0, 100, [1, ]).item()
    stop = np.random.randint(100, 200, [1, ]).item()
    step = np.random.randint(0, 50, [1, ]).item()
    cSlice = NumCpp.Slice(start, stop, step)
    assert cSlice.start == start
    assert cSlice.stop == stop
    assert cSlice.step == step

    cSlice2 = NumCpp.Slice(cSlice)
    assert cSlice2.start == cSlice.start
    assert cSlice2.stop == cSlice.stop
    assert cSlice2.step == cSlice.step

    start = np.random.randint(0, 100, [1, ]).item()
    stop = np.random.randint(100, 200, [1, ]).item()
    step = np.random.randint(0, 50, [1, ]).item()
    cSlice = NumCpp.Slice()
    cSlice.start = start
    cSlice.stop = stop
    cSlice.step = step
    assert cSlice.start == start and cSlice.stop == stop and cSlice.step == step

    cSlice.print()
