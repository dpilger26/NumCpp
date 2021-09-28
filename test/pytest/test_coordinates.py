import numpy as np
from astropy.coordinates import SkyCoord
from astropy.coordinates import Latitude, Longitude  # Angles
import astropy.units as u
import sys
import os
sys.path.append(os.path.abspath(r'../lib'))
import NumCppPy as NumCpp  # noqa E402

np.random.seed(666)


####################################################################################
def test_ra_default_constructor():
    ra = NumCpp.Ra()
    assert ra


####################################################################################
def test_ra_degrees_constructor():
    randDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(randDegrees)
    raPy = Longitude(randDegrees, unit=u.deg)  # noqa
    assert round(ra.degrees(), 9) == round(randDegrees, 9)
    assert ra.hours() == raPy.hms.h
    assert ra.minutes() == raPy.hms.m
    assert round(ra.seconds(), 9) == round(raPy.hms.s, 9)
    assert round(ra.radians(), 9) == round(np.deg2rad(randDegrees), 9)


####################################################################################
def test_ra_hms_constructor():
    hours = np.random.randint(0, 24, [1, ], dtype=np.uint8).item()
    minutes = np.random.randint(0, 60, [1, ], dtype=np.uint8).item()
    seconds = np.random.rand(1).astype(float).item() * 60
    ra = NumCpp.Ra(hours, minutes, seconds)
    degreesPy = (hours + minutes / 60 + seconds / 3600) * 15
    assert round(ra.degrees(), 9) == round(degreesPy, 9)
    assert ra.hours() == hours
    assert ra.minutes() == minutes
    assert round(ra.seconds(), 9) == round(seconds, 9)
    assert round(ra.radians(), 9) == round(np.deg2rad(degreesPy), 9)


####################################################################################
def test_ra_copy_constructor():
    ra = NumCpp.Ra()
    assert ra

    ra2 = NumCpp.Ra(ra)
    assert ra == ra2


####################################################################################
def test_ra_equality_operator():
    ra = NumCpp.Ra()
    assert ra

    randDegrees = np.random.rand(1).item() * 360
    ra2 = NumCpp.Ra(randDegrees)
    assert ra != ra2


####################################################################################
def test_ra_print():
    ra = NumCpp.Ra()
    ra.print()


####################################################################################
def test_dec_default_constructor():
    dec = NumCpp.Dec()
    assert dec


####################################################################################
def test_dec_degree_constructor():
    randDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(randDegrees)
    decPy = Latitude(randDegrees, unit=u.deg)  # noqa
    sign = NumCpp.Sign.NEGATIVE if randDegrees < 0 else NumCpp.Sign.POSITIVE
    assert round(dec.degrees(), 8) == round(randDegrees, 8)
    assert dec.sign() == sign
    assert dec.degreesWhole() == abs(decPy.dms.d)
    assert dec.minutes() == abs(decPy.dms.m)
    assert round(dec.seconds(), 8) == round(abs(decPy.dms.s), 8)
    assert round(dec.radians(), 8) == round(np.deg2rad(randDegrees), 8)


####################################################################################
def test_dec_hms_constructor():
    sign = NumCpp.Sign.POSITIVE if np.random.randint(-1, 1) == 0 else NumCpp.Sign.NEGATIVE
    degrees = np.random.randint(0, 91, [1, ], dtype=np.uint8).item()
    minutes = np.random.randint(0, 60, [1, ], dtype=np.uint8).item()
    seconds = np.random.rand(1).astype(float).item() * 60
    dec = NumCpp.Dec(sign, degrees, minutes, seconds)
    degreesPy = degrees + minutes / 60 + seconds / 3600
    if sign == NumCpp.Sign.NEGATIVE:
        degreesPy *= -1
    assert dec.sign() == sign
    assert round(dec.degrees(), 9) == round(degreesPy, 9)
    assert dec.degreesWhole() == degrees
    assert dec.minutes() == minutes
    assert round(dec.seconds(), 9) == round(seconds, 9)
    assert round(dec.radians(), 8) == round(np.deg2rad(degreesPy), 8)


####################################################################################
def test_dec_copy_constructor():
    dec = NumCpp.Dec()
    assert dec

    dec2 = NumCpp.Dec(dec)
    assert dec == dec2


####################################################################################
def test_equality_operator():
    dec = NumCpp.Dec()
    assert dec

    randDegrees = np.random.rand(1).item() * 180 - 90
    dec2 = NumCpp.Dec(randDegrees)
    assert dec != dec2


####################################################################################
def test_dec_print():
    dec = NumCpp.Dec()
    dec.print()


####################################################################################
def test_coord_default_constructor():
    coord = NumCpp.Coordinate()
    assert coord


####################################################################################
def test_coord_degree_constructor():
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)

    pyCoord = SkyCoord(raDegrees, decDegrees, unit=u.deg)  # noqa
    cCoord = NumCpp.Coordinate(raDegrees, decDegrees)
    assert cCoord.ra() == ra
    assert cCoord.dec() == dec
    assert round(cCoord.x(), 10) == round(pyCoord.cartesian.x.value, 10)
    assert round(cCoord.y(), 10) == round(pyCoord.cartesian.y.value, 10)
    assert round(cCoord.z(), 10) == round(pyCoord.cartesian.z.value, 10)


####################################################################################
def test_coord_radec_constructor():
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)

    pyCoord = SkyCoord(raDegrees, decDegrees, unit=u.deg)  # noqa
    cCoord = NumCpp.Coordinate(ra, dec)
    assert cCoord.ra() == ra
    assert cCoord.dec() == dec
    assert round(cCoord.x(), 10) == round(pyCoord.cartesian.x.value, 10)
    assert round(cCoord.y(), 10) == round(pyCoord.cartesian.y.value, 10)
    assert round(cCoord.z(), 10) == round(pyCoord.cartesian.z.value, 10)


####################################################################################
def test_coord_cartesian_constructor():
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)
    pyCoord = SkyCoord(raDegrees, decDegrees, unit=u.deg)  # noqa
    cCoord = NumCpp.Coordinate(pyCoord.cartesian.x.value, pyCoord.cartesian.y.value, pyCoord.cartesian.z.value)
    assert round(cCoord.ra().degrees(), 9) == round(ra.degrees(), 9)
    assert round(cCoord.dec().degrees(), 9) == round(dec.degrees(), 9)
    assert round(cCoord.x(), 9) == round(pyCoord.cartesian.x.value, 9)
    assert round(cCoord.y(), 9) == round(pyCoord.cartesian.y.value, 9)
    assert round(cCoord.z(), 9) == round(pyCoord.cartesian.z.value, 9)


####################################################################################
def test_coord_cartesian_vector_constructor():
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)
    pyCoord = SkyCoord(raDegrees, decDegrees, unit=u.deg)  # noqa
    vec = np.asarray([pyCoord.cartesian.x.value, pyCoord.cartesian.y.value, pyCoord.cartesian.z.value])
    cVec = NumCpp.NdArray(1, 3)
    cVec.setArray(vec)
    cCoord = NumCpp.Coordinate(cVec)
    assert round(cCoord.ra().degrees(), 9) == round(ra.degrees(), 9)
    assert round(cCoord.dec().degrees(), 9) == round(dec.degrees(), 9)
    assert round(cCoord.x(), 9) == round(pyCoord.cartesian.x.value, 9)
    assert round(cCoord.y(), 9) == round(pyCoord.cartesian.y.value, 9)
    assert round(cCoord.z(), 9) == round(pyCoord.cartesian.z.value, 9)


####################################################################################
def test_coord_rms_constructor():
    raHours = np.random.randint(0, 24, [1, ], dtype=np.uint8).item()
    raMinutes = np.random.randint(0, 60, [1, ], dtype=np.uint8).item()
    raSeconds = np.random.rand(1).astype(float).item() * 60
    raDegreesPy = (raHours + raMinutes / 60 + raSeconds / 3600) * 15

    decSign = NumCpp.Sign.POSITIVE if np.random.randint(-1, 1) == 0 else NumCpp.Sign.NEGATIVE
    decDegrees = np.random.randint(0, 90, [1, ], dtype=np.uint8).item()
    decMinutes = np.random.randint(0, 60, [1, ], dtype=np.uint8).item()
    decSeconds = np.random.rand(1).astype(float).item() * 60
    decDegreesPy = decDegrees + decMinutes / 60 + decSeconds / 3600
    if decSign == NumCpp.Sign.NEGATIVE:
        decDegreesPy *= -1

    cCoord = NumCpp.Coordinate(raHours, raMinutes, raSeconds, decSign, decDegrees, decMinutes, decSeconds)
    cRa = cCoord.ra()
    cDec = cCoord.dec()
    pyCoord = SkyCoord(raDegreesPy, decDegreesPy, unit=u.deg)  # noqa
    assert round(cRa.degrees(), 9) == round(raDegreesPy, 9)
    assert cRa.hours() == raHours
    assert cRa.minutes() == raMinutes
    assert round(cRa.seconds(), 9) == round(raSeconds, 9)
    assert cDec.sign() == decSign
    assert round(cDec.degrees(), 9) == round(decDegreesPy, 9)
    assert cDec.degreesWhole() == decDegrees
    assert cDec.minutes() == decMinutes
    assert round(cDec.seconds(), 9) == round(decSeconds, 9)
    assert round(cCoord.x(), 9) == round(pyCoord.cartesian.x.value, 9)
    assert round(cCoord.y(), 9) == round(pyCoord.cartesian.y.value, 9)
    assert round(cCoord.z(), 9) == round(pyCoord.cartesian.z.value, 9)


####################################################################################
def test_coord_copy_constructor_and_equality_operator():
    cCoord = NumCpp.Coordinate()
    assert cCoord
    cCoord2 = NumCpp.Coordinate(cCoord)
    assert cCoord2 == cCoord


####################################################################################
def test_coord_not_equality_operator():
    cCoord = NumCpp.Coordinate()

    raDegrees = np.random.rand(1).item() * 360
    decDegrees = np.random.rand(1).item() * 180 - 90
    cCoord2 = NumCpp.Coordinate(raDegrees, decDegrees)
    assert cCoord2 != cCoord


####################################################################################
def test_coord_xyz():
    cCoord = NumCpp.Coordinate()
    xyz = [cCoord.x(), cCoord.y(), cCoord.z()]
    assert np.array_equal(cCoord.xyz().getNumpyArray().flatten(), xyz)


####################################################################################
def test_coord_degreeSeperation():
    cCoord = NumCpp.Coordinate()

    raDegrees = np.random.rand(1).item() * 360
    decDegrees = np.random.rand(1).item() * 180 - 90
    cCoord2 = NumCpp.Coordinate(raDegrees, decDegrees)

    pyCoord = SkyCoord(cCoord.ra().degrees(), cCoord.dec().degrees(), unit=u.deg)  # noqa
    pyCoord2 = SkyCoord(cCoord2.ra().degrees(), cCoord2.dec().degrees(), unit=u.deg)  # noqa

    cDegSep = cCoord.degreeSeperation(cCoord2)
    pyDegSep = pyCoord.separation(pyCoord2).value
    assert round(cDegSep, 9) == round(pyDegSep, 9)


####################################################################################
def test_coord_radianSeperation():
    cCoord = NumCpp.Coordinate()

    raDegrees = np.random.rand(1).item() * 360
    decDegrees = np.random.rand(1).item() * 180 - 90
    cCoord2 = NumCpp.Coordinate(raDegrees, decDegrees)

    pyCoord = SkyCoord(cCoord.ra().degrees(), cCoord.dec().degrees(), unit=u.deg)  # noqa
    pyCoord2 = SkyCoord(cCoord2.ra().degrees(), cCoord2.dec().degrees(), unit=u.deg)  # noqa

    cRadSep = cCoord.radianSeperation(cCoord2)
    pyRadSep = np.deg2rad(pyCoord.separation(pyCoord2).value)

    assert round(cRadSep, 9) == round(pyRadSep, 9)


####################################################################################
def test_coord_degreeSeperation_vec():
    cCoord = NumCpp.Coordinate()

    raDegrees = np.random.rand(1).item() * 360
    decDegrees = np.random.rand(1).item() * 180 - 90
    cCoord2 = NumCpp.Coordinate(raDegrees, decDegrees)

    pyCoord = SkyCoord(cCoord.ra().degrees(), cCoord.dec().degrees(), unit=u.deg)  # noqa
    pyCoord2 = SkyCoord(cCoord2.ra().degrees(), cCoord2.dec().degrees(), unit=u.deg)  # noqa

    vec2 = np.asarray([pyCoord2.cartesian.x, pyCoord2.cartesian.y, pyCoord2.cartesian.z])
    cArray = NumCpp.NdArray(1, 3)
    cArray.setArray(vec2)
    cDegSep = cCoord.degreeSeperation(cArray)
    pyDegSep = pyCoord.separation(pyCoord2).value
    assert round(cDegSep, 9) == round(pyDegSep, 9)


####################################################################################
def test_coord_radianSeperation_vec():
    cCoord = NumCpp.Coordinate()

    raDegrees = np.random.rand(1).item() * 360
    decDegrees = np.random.rand(1).item() * 180 - 90
    cCoord2 = NumCpp.Coordinate(raDegrees, decDegrees)

    pyCoord = SkyCoord(cCoord.ra().degrees(), cCoord.dec().degrees(), unit=u.deg)  # noqa
    pyCoord2 = SkyCoord(cCoord2.ra().degrees(), cCoord2.dec().degrees(), unit=u.deg)  # noqa

    vec2 = np.asarray([pyCoord2.cartesian.x, pyCoord2.cartesian.y, pyCoord2.cartesian.z])
    cArray = NumCpp.NdArray(1, 3)
    cArray.setArray(vec2)
    cRadSep = cCoord.radianSeperation(cArray)
    pyRadSep = np.radians(pyCoord.separation(pyCoord2).value)
    assert round(cRadSep, 9) == round(pyRadSep, 9)  # noqa


####################################################################################
def test_coorc_print():
    cCoord = NumCpp.Coordinate()
    cCoord.print()
