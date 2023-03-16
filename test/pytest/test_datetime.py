from typing import Iterable, Callable

import numpy as np

import NumCppPy as NumCpp  # noqa E402


####################################################################################
def test_datetime():
    if NumCpp.NUMCPP_NO_USE_BOOST:
        return

    d = NumCpp.DateTime()
    assert d.year == 1970
    assert d.month == 1
    assert d.day == 1
    assert d.hour == 0
    assert d.minute == 0
    assert d.second == 0
    assert d.fractionalSecond == 0.0
    assert d.toStr() == "1970-01-01T00:00:00Z"

    year = 2000
    month = 2
    day = 3
    hour = 4
    minute = 5
    second = 6
    fractionalSecond = 0.7

    d = NumCpp.DateTime(year, month, day, hour, minute,
                        second, fractionalSecond)
    assert d.year == year
    assert d.month == month
    assert d.day == day
    assert d.hour == hour
    assert d.minute == minute
    assert d.second == second
    assert d.fractionalSecond == fractionalSecond
    assert d.toStr() == "2000-02-03T04:05:06.7Z"

    d = NumCpp.DateTime()
    d.year = year
    d.month = month
    d.day = day
    d.hour = hour
    d.minute = minute
    d.second = second
    d.fractionalSecond = fractionalSecond
    assert d.year == year
    assert d.month == month
    assert d.day == day
    assert d.hour == hour
    assert d.minute == minute
    assert d.second == second
    assert d.fractionalSecond == fractionalSecond
    assert d.toStr() == "2000-02-03T04:05:06.7Z"

    d = NumCpp.DateTime.now()
    tp = d.toTimePoint()
    assert tp.time_since_epoch().count() > 0

    d2 = NumCpp.DateTime.now()

    assert not d == d2
    assert d != d2
    assert d < d2
    assert d <= d2
    assert not d > d2
    assert not d >= d2
    assert (d2 - d) > NumCpp.Duration(0)

    def assertThrow(expression: Callable, inputs: Iterable):
        try:
            expression(*inputs)
            assert False, "expression should throw"
        except Exception:  # std::invalid_argument
            assert True

    assertThrow(NumCpp.DateTime, "1-00-29T11:52:33.123456789Z")
    assertThrow(NumCpp.DateTime, "22-00-29T11:52:33.123456789Z")
    assertThrow(NumCpp.DateTime, "222-00-29T11:52:33.123456789Z")
    assertThrow(NumCpp.DateTime, "2022-0-29T11:52:33.123456789Z")
    assertThrow(NumCpp.DateTime, "2022-00-29T11:52:33.123456789Z")
    assertThrow(NumCpp.DateTime, "2022-13-29T11:52:33.123456789Z")
    assertThrow(NumCpp.DateTime, "2022-11-0T11:52:33.123456789Z")
    assertThrow(NumCpp.DateTime, "2022-11-00T11:52:33.123456789Z")
    assertThrow(NumCpp.DateTime, "2022-11-32T11:52:33.123456789Z")
    assertThrow(NumCpp.DateTime, "2022-11-29T24:52:33.123456789Z")
    assertThrow(NumCpp.DateTime, "2022-11-29T11:60:33.123456789Z")
    assertThrow(NumCpp.DateTime, "2022-11-29T11:52:60.123456789Z")
    assertThrow(NumCpp.DateTime, "2022-11-29T11:52:33.123456789")

    timestamp = "2022-11-29T11:52:33.123456Z"
    d = NumCpp.DateTime(timestamp)
    assert d.year == 2022
    assert d.month == 11
    assert d.day == 29
    assert d.hour == 11
    assert d.minute == 52
    assert d.second == 33
    assert d.fractionalSecond == 0.123456
    assert d.toStr() == timestamp

    # this is pretty cool
    timestamp2 = "2022-11-29T24:52:33.123456Z"
    d = NumCpp.DateTime(timestamp2)
    assert d.year == 2022
    assert d.month == 11
    assert d.day == 30
    assert d.hour == 0
    assert d.minute == 52
    assert d.second == 33
    assert d.fractionalSecond == 0.123456
    assert d.toStr() == "2022-11-30T00:52:33.123456Z"

    tp2 = NumCpp.Clock.now()
    d = NumCpp.DateTime(tp2)
    assert np.abs(d.toTimePoint().time_since_epoch().count() - tp2.time_since_epoch().count()) <= 1
