import NumCppPy as NumCpp  # noqa E402


def test_logger():
    if NumCpp.NUMCPP_NO_USE_BOOST:
        return


def test_binaryLogger():
    if NumCpp.NUMCPP_NO_USE_BOOST:
        return
