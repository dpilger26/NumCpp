import numpy as np

import NumCppPy as NumCpp  # noqa E402

DISABLE_PRINTS = True


####################################################################################
def test_default_constructor():
    cSlice = NumCpp.Slice()
    assert cSlice.start == 0
    assert cSlice.stop == 1
    assert cSlice.step == 1


####################################################################################
def test_stop_constructor():
    stop = np.random.randint(
        0,
        100,
        [
            1,
        ],
    ).item()
    cSlice = NumCpp.Slice(stop)
    assert cSlice.start == 0
    assert cSlice.stop == stop
    assert cSlice.step == 1


####################################################################################
def test_start_stop_constructor():
    start = np.random.randint(
        0,
        100,
        [
            1,
        ],
    ).item()
    stop = np.random.randint(
        100,
        200,
        [
            1,
        ],
    ).item()
    cSlice = NumCpp.Slice(start, stop)
    assert cSlice.start == start
    assert cSlice.stop == stop
    assert cSlice.step == 1


####################################################################################
def test_start_stop_step_constructor():
    start = np.random.randint(
        0,
        100,
        [
            1,
        ],
    ).item()
    stop = np.random.randint(
        100,
        200,
        [
            1,
        ],
    ).item()
    step = np.random.randint(
        0,
        50,
        [
            1,
        ],
    ).item()
    cSlice = NumCpp.Slice(start, stop, step)
    assert cSlice.start == start
    assert cSlice.stop == stop
    assert cSlice.step == step


####################################################################################
def test_copy_constructor():
    start = np.random.randint(
        0,
        100,
        [
            1,
        ],
    ).item()
    stop = np.random.randint(
        100,
        200,
        [
            1,
        ],
    ).item()
    step = np.random.randint(
        0,
        50,
        [
            1,
        ],
    ).item()
    cSlice = NumCpp.Slice(start, stop, step)

    cSlice2 = NumCpp.Slice(cSlice)
    assert cSlice2.start == cSlice.start
    assert cSlice2.stop == cSlice.stop
    assert cSlice2.step == cSlice.step


####################################################################################
def test_set():
    start = np.random.randint(
        0,
        100,
        [
            1,
        ],
    ).item()
    stop = np.random.randint(
        100,
        200,
        [
            1,
        ],
    ).item()
    step = np.random.randint(
        0,
        50,
        [
            1,
        ],
    ).item()
    cSlice = NumCpp.Slice()
    cSlice.start = start
    cSlice.stop = stop
    cSlice.step = step
    assert cSlice.start == start and cSlice.stop == stop and cSlice.step == step

    if not DISABLE_PRINTS:
        cSlice.print()


####################################################################################
def test_numElements():
    arraySize = 300
    for _ in range(100):
        start = np.random.randint(
            0,
            arraySize // 2,
            [
                1,
            ],
        ).item()
        stop = np.random.randint(
            arraySize // 2,
            arraySize,
            [
                1,
            ],
        ).item()
        step = np.random.randint(
            2,
            5,
            [
                1,
            ],
        ).item()
        cSlice = NumCpp.Slice(start, stop, step)
        indices = np.arange(start, stop, step)
        assert cSlice.numElements(arraySize) == indices.size


####################################################################################
def test_toIndices():
    arraySize = 300
    for _ in range(100):
        start = np.random.randint(
            0,
            arraySize // 2,
            [
                1,
            ],
        ).item()
        stop = np.random.randint(
            arraySize // 2,
            arraySize,
            [
                1,
            ],
        ).item()
        step = np.random.randint(
            2,
            5,
            [
                1,
            ],
        ).item()
        cSlice = NumCpp.Slice(start, stop, step)
        indices = np.arange(start, stop, step)
        assert np.array_equal(cSlice.toIndices(arraySize), indices)
