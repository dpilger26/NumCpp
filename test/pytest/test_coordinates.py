import numpy as np
from astropy.coordinates import SkyCoord
from astropy.coordinates import Latitude, Longitude  # Angles
import astropy.units as u

import NumCppPy as NumCpp  # noqa E402

np.random.seed(666)


####################################################################################
def test_cartesian_default_constructor():
    c = NumCpp.Cartesian()
    assert c.x == 0
    assert c.y == 0
    assert c.z == 0


####################################################################################
def test_cartesian_component_constructor():
    x, y, z = np.random.rand(3) * 10
    c = NumCpp.Cartesian(x, y, z)
    assert c.x == x
    assert c.y == y
    assert c.z == z


####################################################################################
def test_cartesian_vec2_constructor():
    x, y = np.random.rand(2) * 10
    vec2 = NumCpp.Vec2(x, y)
    c = NumCpp.Cartesian(vec2)
    assert c.x == x
    assert c.y == y
    assert c.z == 0


####################################################################################
def test_cartesian_vec3_constructor():
    x, y, z = np.random.rand(3) * 10
    vec3 = NumCpp.Vec3(x, y, z)
    c = NumCpp.Cartesian(vec3)
    assert c.x == x
    assert c.y == y
    assert c.z == z


####################################################################################
def test_cartesian_ndarray_constructor():
    components = np.random.rand(3) * 10
    cArray = NumCpp.NdArray(3, 1)
    cArray.setArray(components)
    c = NumCpp.Cartesian(cArray)
    assert c.x == components[0]
    assert c.y == components[1]
    assert c.z == components[2]


####################################################################################
def test_cartesian_xHat():
    c = NumCpp.Cartesian.xHat()
    assert c.x == 1
    assert c.y == 0
    assert c.z == 0


####################################################################################
def test_cartesian_yHat():
    c = NumCpp.Cartesian.yHat()
    assert c.x == 0
    assert c.y == 1
    assert c.z == 0


####################################################################################
def test_cartesian_zHat():
    c = NumCpp.Cartesian.zHat()
    assert c.x == 0
    assert c.y == 0
    assert c.z == 1


####################################################################################
def test_cartesian_eq():
    c1 = NumCpp.Cartesian.xHat()
    c2 = NumCpp.Cartesian.xHat()
    assert c1 == c2


####################################################################################
def test_cartesian_ne():
    c1 = NumCpp.Cartesian.xHat()
    c2 = NumCpp.Cartesian.zHat()
    assert c1 != c2


####################################################################################
def test_cartesian_add():
    c1 = NumCpp.Cartesian.xHat()
    c2 = NumCpp.Cartesian.zHat()
    c = c1 + c2
    assert c.x == 1
    assert c.y == 0
    assert c.z == 1


####################################################################################
def test_cartesian_sub():
    c1 = NumCpp.Cartesian.xHat()
    c2 = NumCpp.Cartesian.zHat()
    c = c1 - c2
    assert c.x == 1
    assert c.y == 0
    assert c.z == -1


####################################################################################
def test_cartesian_mul():
    c = NumCpp.Cartesian.xHat() * 10
    assert c.x == 10
    assert c.y == 0
    assert c.z == 0


####################################################################################
def test_cartesian_div():
    c = NumCpp.Cartesian.xHat() / 10
    assert c.x == 0.1
    assert c.y == 0
    assert c.z == 0


####################################################################################
def test_cartesian_print():
    c = NumCpp.Cartesian.xHat()
    c.print()


####################################################################################
def test_cartesian_cross():
    c1 = NumCpp.Cartesian.xHat()
    c2 = NumCpp.Cartesian.yHat()
    c = NumCpp.cross(c1, c2)
    assert c == NumCpp.Cartesian.zHat()


####################################################################################
def test_cartesian_norm():
    x, y, z = np.random.rand(3) * 100
    c = NumCpp.Cartesian(x, y, z)
    assert np.round(NumCpp.norm(c), 9) == np.round(np.linalg.norm([x, y, z]), 9)


####################################################################################
def test_cartesian_normalize():
    x, y, z = np.random.rand(3) * 100
    c = NumCpp.normalize(NumCpp.Cartesian(x, y, z))
    assert NumCpp.norm(c) == 1


####################################################################################
def test_cartesian_angle():
    x1, y1, z1 = np.random.rand(3) * 100
    x2, y2, z2 = np.random.rand(3) * 100
    c1 = NumCpp.Cartesian(x1, y1, z1)
    c2 = NumCpp.Cartesian(x2, y2, z2)

    a1 = np.array([x1, y1, z1])
    a1 = a1 / np.linalg.norm(a1)
    a2 = np.array([x2, y2, z2])
    a2 = a2 / np.linalg.norm(a2)
    angle = np.arccos(np.dot(a1, a2))

    assert np.round(NumCpp.angle(c1, c2), 9) == np.round(angle, 9)


####################################################################################
def test_ra_default_constructor():
    ra = NumCpp.Ra()
    assert ra


####################################################################################
def test_ra_degrees_constructor():
    randDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(randDegrees)
    raPy = Longitude(randDegrees, unit=u.deg)  # noqa
    assert round(ra.degrees(), 8) == round(randDegrees, 8)
    assert ra.hours() == raPy.hms.h
    assert ra.minutes() == raPy.hms.m
    assert round(ra.seconds(), 8) == round(raPy.hms.s, 8)
    assert round(ra.radians(), 8) == round(np.deg2rad(randDegrees), 8)


####################################################################################
def test_ra_hms_constructor():
    hours = np.random.randint(
        0,
        24,
        [
            1,
        ],
        dtype=np.uint8,
    ).item()
    minutes = np.random.randint(
        0,
        60,
        [
            1,
        ],
        dtype=np.uint8,
    ).item()
    seconds = np.random.rand(1).astype(float).item() * 60
    ra = NumCpp.Ra(hours, minutes, seconds)
    degreesPy = (hours + minutes / 60 + seconds / 3600) * 15
    assert round(ra.degrees(), 8) == round(degreesPy, 8)
    assert ra.hours() == hours
    assert ra.minutes() == minutes
    assert round(ra.seconds(), 8) == round(seconds, 8)
    assert round(ra.radians(), 8) == round(np.deg2rad(degreesPy), 8)


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
    degrees = np.random.randint(
        0,
        91,
        [
            1,
        ],
        dtype=np.uint8,
    ).item()
    minutes = np.random.randint(
        0,
        60,
        [
            1,
        ],
        dtype=np.uint8,
    ).item()
    seconds = np.random.rand(1).astype(float).item() * 60
    dec = NumCpp.Dec(sign, degrees, minutes, seconds)
    degreesPy = degrees + minutes / 60 + seconds / 3600
    if sign == NumCpp.Sign.NEGATIVE:
        degreesPy *= -1
    assert dec.sign() == sign
    assert round(dec.degrees(), 8) == round(degreesPy, 8)
    assert dec.degreesWhole() == degrees
    assert dec.minutes() == minutes
    assert round(dec.seconds(), 8) == round(seconds, 8)
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
    coord = NumCpp.Celestial()
    assert coord


####################################################################################
def test_coord_degree_constructor():
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)

    pyCoord = SkyCoord(raDegrees, decDegrees, unit=u.deg)  # noqa
    cCoord = NumCpp.Celestial(raDegrees, decDegrees)
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
    cCoord = NumCpp.Celestial(ra, dec)
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
    cCoord = NumCpp.Celestial(pyCoord.cartesian.x.value, pyCoord.cartesian.y.value, pyCoord.cartesian.z.value)
    assert round(cCoord.ra().degrees(), 8) == round(ra.degrees(), 8)
    assert round(cCoord.dec().degrees(), 8) == round(dec.degrees(), 8)
    assert round(cCoord.x(), 8) == round(pyCoord.cartesian.x.value, 8)
    assert round(cCoord.y(), 8) == round(pyCoord.cartesian.y.value, 8)
    assert round(cCoord.z(), 8) == round(pyCoord.cartesian.z.value, 8)


####################################################################################
def test_coord_ndarray_constructor():
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)
    pyCoord = SkyCoord(raDegrees, decDegrees, unit=u.deg)  # noqa
    vec = np.asarray([pyCoord.cartesian.x.value, pyCoord.cartesian.y.value, pyCoord.cartesian.z.value])
    cVec = NumCpp.NdArray(1, 3)
    cVec.setArray(vec)
    cCoord = NumCpp.Celestial(cVec)
    assert round(cCoord.ra().degrees(), 8) == round(ra.degrees(), 8)
    assert round(cCoord.dec().degrees(), 8) == round(dec.degrees(), 8)
    assert round(cCoord.x(), 8) == round(pyCoord.cartesian.x.value, 8)
    assert round(cCoord.y(), 8) == round(pyCoord.cartesian.y.value, 8)
    assert round(cCoord.z(), 8) == round(pyCoord.cartesian.z.value, 8)


####################################################################################
def test_coord_cartesian_constructor():
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)
    pyCoord = SkyCoord(raDegrees, decDegrees, unit=u.deg)  # noqa
    cartesian = NumCpp.Cartesian(pyCoord.cartesian.x.value, pyCoord.cartesian.y.value, pyCoord.cartesian.z.value)
    cCoord = NumCpp.Celestial(cartesian)
    assert round(cCoord.ra().degrees(), 8) == round(ra.degrees(), 8)
    assert round(cCoord.dec().degrees(), 8) == round(dec.degrees(), 8)
    assert round(cCoord.x(), 8) == round(pyCoord.cartesian.x.value, 8)
    assert round(cCoord.y(), 8) == round(pyCoord.cartesian.y.value, 8)
    assert round(cCoord.z(), 8) == round(pyCoord.cartesian.z.value, 8)


####################################################################################
def test_coord_vec3_constructor():
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)
    pyCoord = SkyCoord(raDegrees, decDegrees, unit=u.deg)  # noqa
    vec3 = NumCpp.Vec3(pyCoord.cartesian.x.value, pyCoord.cartesian.y.value, pyCoord.cartesian.z.value)
    cCoord = NumCpp.Celestial(vec3)
    assert round(cCoord.ra().degrees(), 8) == round(ra.degrees(), 8)
    assert round(cCoord.dec().degrees(), 8) == round(dec.degrees(), 8)
    assert round(cCoord.x(), 8) == round(pyCoord.cartesian.x.value, 8)
    assert round(cCoord.y(), 8) == round(pyCoord.cartesian.y.value, 8)
    assert round(cCoord.z(), 8) == round(pyCoord.cartesian.z.value, 8)


####################################################################################
def test_coord_rms_constructor():
    raHours = np.random.randint(
        0,
        24,
        [
            1,
        ],
        dtype=np.uint8,
    ).item()
    raMinutes = np.random.randint(
        0,
        60,
        [
            1,
        ],
        dtype=np.uint8,
    ).item()
    raSeconds = np.random.rand(1).astype(float).item() * 60
    raDegreesPy = (raHours + raMinutes / 60 + raSeconds / 3600) * 15

    decSign = NumCpp.Sign.POSITIVE if np.random.randint(-1, 1) == 0 else NumCpp.Sign.NEGATIVE
    decDegrees = np.random.randint(
        0,
        90,
        [
            1,
        ],
        dtype=np.uint8,
    ).item()
    decMinutes = np.random.randint(
        0,
        60,
        [
            1,
        ],
        dtype=np.uint8,
    ).item()
    decSeconds = np.random.rand(1).astype(float).item() * 60
    decDegreesPy = decDegrees + decMinutes / 60 + decSeconds / 3600
    if decSign == NumCpp.Sign.NEGATIVE:
        decDegreesPy *= -1

    cCoord = NumCpp.Celestial(raHours, raMinutes, raSeconds, decSign, decDegrees, decMinutes, decSeconds)
    cRa = cCoord.ra()
    cDec = cCoord.dec()
    pyCoord = SkyCoord(raDegreesPy, decDegreesPy, unit=u.deg)  # noqa
    assert round(cRa.degrees(), 8) == round(raDegreesPy, 8)
    assert cRa.hours() == raHours
    assert cRa.minutes() == raMinutes
    assert round(cRa.seconds(), 8) == round(raSeconds, 8)
    assert cDec.sign() == decSign
    assert round(cDec.degrees(), 8) == round(decDegreesPy, 8)
    assert cDec.degreesWhole() == decDegrees
    assert cDec.minutes() == decMinutes
    assert round(cDec.seconds(), 8) == round(decSeconds, 8)
    assert round(cCoord.x(), 8) == round(pyCoord.cartesian.x.value, 8)
    assert round(cCoord.y(), 8) == round(pyCoord.cartesian.y.value, 8)
    assert round(cCoord.z(), 8) == round(pyCoord.cartesian.z.value, 8)


####################################################################################
def test_coord_copy_constructor_and_equality_operator():
    cCoord = NumCpp.Celestial()
    assert cCoord
    cCoord2 = NumCpp.Celestial(cCoord)
    assert cCoord2 == cCoord


####################################################################################
def test_coord_not_equality_operator():
    cCoord = NumCpp.Celestial()

    raDegrees = np.random.rand(1).item() * 360
    decDegrees = np.random.rand(1).item() * 180 - 90
    cCoord2 = NumCpp.Celestial(raDegrees, decDegrees)
    assert cCoord2 != cCoord


####################################################################################
def test_coord_xyz():
    cCoord = NumCpp.Celestial()
    xyz = [cCoord.x(), cCoord.y(), cCoord.z()]
    assert np.array_equal(cCoord.xyz().getNumpyArray().flatten(), xyz)


####################################################################################
def test_coord_degreeSeperation():
    cCoord = NumCpp.Celestial()

    raDegrees = np.random.rand(1).item() * 360
    decDegrees = np.random.rand(1).item() * 180 - 90
    cCoord2 = NumCpp.Celestial(raDegrees, decDegrees)

    pyCoord = SkyCoord(cCoord.ra().degrees(), cCoord.dec().degrees(), unit=u.deg)  # noqa
    pyCoord2 = SkyCoord(cCoord2.ra().degrees(), cCoord2.dec().degrees(), unit=u.deg)  # noqa

    cDegSep = cCoord.degreeSeperation(cCoord2)
    pyDegSep = pyCoord.separation(pyCoord2).value
    assert round(cDegSep, 8) == round(pyDegSep, 8)


####################################################################################
def test_coord_radianSeperation():
    cCoord = NumCpp.Celestial()

    raDegrees = np.random.rand(1).item() * 360
    decDegrees = np.random.rand(1).item() * 180 - 90
    cCoord2 = NumCpp.Celestial(raDegrees, decDegrees)

    pyCoord = SkyCoord(cCoord.ra().degrees(), cCoord.dec().degrees(), unit=u.deg)  # noqa
    pyCoord2 = SkyCoord(cCoord2.ra().degrees(), cCoord2.dec().degrees(), unit=u.deg)  # noqa

    cRadSep = cCoord.radianSeperation(cCoord2)
    pyRadSep = np.deg2rad(pyCoord.separation(pyCoord2).value)

    assert round(cRadSep, 8) == round(pyRadSep, 8)


####################################################################################
def test_coord_degreeSeperation_vec():
    cCoord = NumCpp.Celestial()

    raDegrees = np.random.rand(1).item() * 360
    decDegrees = np.random.rand(1).item() * 180 - 90
    cCoord2 = NumCpp.Celestial(raDegrees, decDegrees)

    pyCoord = SkyCoord(cCoord.ra().degrees(), cCoord.dec().degrees(), unit=u.deg)  # noqa
    pyCoord2 = SkyCoord(cCoord2.ra().degrees(), cCoord2.dec().degrees(), unit=u.deg)  # noqa

    vec2 = np.asarray([pyCoord2.cartesian.x, pyCoord2.cartesian.y, pyCoord2.cartesian.z])
    cArray = NumCpp.NdArray(1, 3)
    cArray.setArray(vec2)
    cDegSep = cCoord.degreeSeperation(cArray)
    pyDegSep = pyCoord.separation(pyCoord2).value
    assert round(cDegSep, 8) == round(pyDegSep, 8)


####################################################################################
def test_coord_radianSeperation_vec():
    cCoord = NumCpp.Celestial()

    raDegrees = np.random.rand(1).item() * 360
    decDegrees = np.random.rand(1).item() * 180 - 90
    cCoord2 = NumCpp.Celestial(raDegrees, decDegrees)

    pyCoord = SkyCoord(cCoord.ra().degrees(), cCoord.dec().degrees(), unit=u.deg)  # noqa
    pyCoord2 = SkyCoord(cCoord2.ra().degrees(), cCoord2.dec().degrees(), unit=u.deg)  # noqa

    vec2 = np.asarray([pyCoord2.cartesian.x, pyCoord2.cartesian.y, pyCoord2.cartesian.z])
    cArray = NumCpp.NdArray(1, 3)
    cArray.setArray(vec2)
    cRadSep = cCoord.radianSeperation(cArray)
    pyRadSep = np.radians(pyCoord.separation(pyCoord2).value)
    assert round(cRadSep, 8) == round(pyRadSep, 8)  # noqa


####################################################################################
def test_coorc_print():
    cCoord = NumCpp.Celestial()
    cCoord.print()
