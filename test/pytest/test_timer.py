import numpy as np

import NumCppPy as NumCpp  # noqa E402


####################################################################################
def test_timer():
    np.random.seed(666)

    """Tests the NumCpp Timer class"""
    SLEEP_TIME = int(
        np.random.randint(
            0,
            10,
            [
                1,
            ],
        ).item()
        * 1e6
    )  # microseconds
    timer = NumCpp.Timer()
    timer.tic()
    timer.sleep(SLEEP_TIME)
    elapsedTime = timer.toc(NumCpp.PrintElapsedTime.YES)  # microseconds
    assert np.abs(elapsedTime.count() - SLEEP_TIME) < 0.1e6

    SLEEP_TIME = int(
        np.random.randint(
            0,
            10,
            [
                1,
            ],
        ).item()
        * 1e6
    )  # microseconds
    timer = NumCpp.Timer("Python Test Case")
    timer.tic()
    timer.sleep(SLEEP_TIME)
    elapsedTime = timer.toc(NumCpp.PrintElapsedTime.YES)  # microseconds
    assert np.abs(elapsedTime.count() - SLEEP_TIME) < 0.1e6
