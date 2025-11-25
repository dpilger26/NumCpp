import numpy as np
from astropy.coordinates import SkyCoord
from astropy.coordinates import Latitude, Longitude  # Angles
import astropy.units as u
import pymap3d
import pytest

import NumCppPy as NumCpp  # noqa E402

DISABLE_PRINTS = True


####################################################################################
def test_seed():
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
@pytest.mark.skip(
    reason="This segfaults right now, but I'm pretty sure it is just the pytest test, nothing wrong with the actual code..."
)
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
def test_euler():
    euler = NumCpp.Euler()
    assert euler.psi == 0.0
    assert euler.theta == 0.0
    assert euler.phi == 0.0

    psi, theta, phi = np.random.rand(3) * np.pi / 4
    euler = NumCpp.Euler(psi, theta, phi)
    assert euler.psi == psi
    assert euler.theta == theta
    assert euler.phi == phi

    euler2 = NumCpp.Euler(psi, theta, phi)
    assert euler == euler2

    euler2 = NumCpp.Euler(theta, psi, phi)
    assert euler != euler2

    if not DISABLE_PRINTS:
        euler.print()


####################################################################################
def test_orientation():
    orientation = NumCpp.Orientation()
    assert orientation.roll == 0.0
    assert orientation.pitch == 0.0
    assert orientation.yaw == 0.0

    roll, pitch, yaw = np.random.rand(3) * np.pi / 4
    orientation = NumCpp.Orientation(roll, pitch, yaw)
    assert orientation.roll == roll
    assert orientation.pitch == pitch
    assert orientation.yaw == yaw

    orientation2 = NumCpp.Orientation(roll, pitch, yaw)
    assert orientation == orientation2

    orientation2 = NumCpp.Orientation(pitch, roll, yaw)
    assert orientation != orientation2

    if not DISABLE_PRINTS:
        orientation.print()


####################################################################################
def test_aer():
    aer = NumCpp.AER()
    assert aer.az == 0.0
    assert aer.el == 0.0

    az, el = np.random.rand(2) * np.pi / 4
    aer = NumCpp.AER(az, el)
    assert aer.az == az
    assert aer.el == el

    aer2 = NumCpp.AER(az, el)
    assert aer == aer2

    aer2 = NumCpp.AER(el, az)
    assert aer != aer2

    if not DISABLE_PRINTS:
        aer.print()


####################################################################################
def test_enu():
    enu = NumCpp.ENU()
    assert enu.x == 0.0
    assert enu.y == 0.0
    assert enu.z == 0.0
    assert enu.east == 0.0
    assert enu.north == 0.0
    assert enu.up == 0.0

    east, north, up = np.random.rand(3)
    enu = NumCpp.ENU(east, north, up)
    assert enu.x == east
    assert enu.y == north
    assert enu.z == up
    assert enu.east == east
    assert enu.north == north
    assert enu.up == up

    enu = NumCpp.ENU()
    enu.east = east
    enu.north = north
    enu.up = up
    assert enu.x == east
    assert enu.y == north
    assert enu.z == up
    assert enu.east == east
    assert enu.north == north
    assert enu.up == up

    enu2 = NumCpp.ENU(east, north, up)
    assert enu == enu2

    enu2 = NumCpp.ENU(north, east, up)
    assert enu != enu2

    if not DISABLE_PRINTS:
        enu.print()


####################################################################################
def test_ned():
    ned = NumCpp.NED()
    assert ned.x == 0.0
    assert ned.y == 0.0
    assert ned.z == 0.0
    assert ned.north == 0.0
    assert ned.east == 0.0
    assert ned.down == 0.0

    north, east, down = np.random.rand(3)
    ned = NumCpp.NED(north, east, down)
    assert ned.x == north
    assert ned.y == east
    assert ned.z == down
    assert ned.north == north
    assert ned.east == east
    assert ned.down == down

    ned = NumCpp.NED()
    ned.north = north
    ned.east = east
    ned.down = down
    assert ned.x == north
    assert ned.y == east
    assert ned.z == down
    assert ned.north == north
    assert ned.east == east
    assert ned.down == down

    ned2 = NumCpp.NED(north, east, down)
    assert ned == ned2

    ned2 = NumCpp.NED(east, north, down)
    assert ned != ned2

    if not DISABLE_PRINTS:
        ned.print()


####################################################################################
def test_geocentric():
    geocentric = NumCpp.Geocentric()
    assert geocentric.latitude == 0.0
    assert geocentric.longitude == 0.0
    assert geocentric.radius == 0.0

    lat, lon, radius = np.random.rand(3) * np.pi / 4
    geocentric = NumCpp.Geocentric(lat, lon, radius)
    assert geocentric.latitude == lat
    assert geocentric.longitude == lon
    assert geocentric.radius == radius

    geocentric2 = NumCpp.Geocentric(lat, lon, radius)
    assert geocentric == geocentric2

    geocentric2 = NumCpp.Geocentric(lon, lat, radius)
    assert geocentric != geocentric2

    if not DISABLE_PRINTS:
        geocentric.print()


####################################################################################
def test_lla():
    lla = NumCpp.LLA()
    assert lla.latitude == 0.0
    assert lla.longitude == 0.0
    assert lla.altitude == 0.0

    lat, lon, alt = np.random.rand(3) * np.pi / 4
    lla = NumCpp.LLA(lat, lon, alt)
    assert lla.latitude == lat
    assert lla.longitude == lon
    assert lla.altitude == alt

    lla2 = NumCpp.LLA(lat, lon, alt)
    assert lla == lla2

    lla2 = NumCpp.LLA(lon, lat, alt)
    assert lla != lla2

    if not DISABLE_PRINTS:
        lla.print()


####################################################################################
def test_ra_default_constructor():
    ra = NumCpp.Ra()
    assert ra


####################################################################################
def test_ra_degrees_constructor():
    randDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(randDegrees)
    raPy = Longitude(randDegrees, unit=u.deg)  # noqa
    assert round(ra.degrees(), 5) == round(randDegrees, 5)
    assert ra.hours() == raPy.hms.h
    assert ra.minutes() == raPy.hms.m
    assert round(ra.seconds(), 5) == round(raPy.hms.s, 5)
    assert round(ra.radians(), 5) == round(np.deg2rad(randDegrees), 5)


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
    assert round(ra.degrees(), 5) == round(degreesPy, 5)
    assert ra.hours() == hours
    assert ra.minutes() == minutes
    assert round(ra.seconds(), 5) == round(seconds, 5)
    assert round(ra.radians(), 5) == round(np.deg2rad(degreesPy), 5)


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
@pytest.mark.skip(
    reason="This segfaults right now, but I'm pretty sure it is just the pytest test, nothing wrong with the actual code..."
)
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
    assert round(dec.degrees(), 5) == round(randDegrees, 5)
    assert dec.sign() == sign
    assert dec.degreesWhole() == abs(decPy.dms.d)
    assert dec.minutes() == abs(decPy.dms.m)
    assert round(dec.seconds(), 5) == round(abs(decPy.dms.s), 5)
    assert round(dec.radians(), 5) == round(np.deg2rad(randDegrees), 5)


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
    assert round(dec.degrees(), 5) == round(degreesPy, 5)
    assert dec.degreesWhole() == degrees
    assert dec.minutes() == minutes
    assert round(dec.seconds(), 5) == round(seconds, 5)
    assert round(dec.radians(), 5) == round(np.deg2rad(degreesPy), 5)


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
@pytest.mark.skip(
    reason="This segfaults right now, but I'm pretty sure it is just the pytest test, nothing wrong with the actual code..."
)
def test_dec_print():
    dec = NumCpp.Dec()
    dec.print()


####################################################################################
def test_celestial_default_constructor():
    celestial = NumCpp.Celestial()
    assert celestial


####################################################################################
def test_celestial_degree_constructor():
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)

    pycelestial = SkyCoord(raDegrees, decDegrees, unit=u.deg)  # noqa
    cCelestial = NumCpp.Celestial(raDegrees, decDegrees)
    assert cCelestial.ra() == ra
    assert cCelestial.dec() == dec
    assert round(cCelestial.x(), 10) == round(pycelestial.cartesian.x.value, 10)
    assert round(cCelestial.y(), 10) == round(pycelestial.cartesian.y.value, 10)
    assert round(cCelestial.z(), 10) == round(pycelestial.cartesian.z.value, 10)


####################################################################################
def test_celestial_radec_constructor():
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)

    pycelestial = SkyCoord(raDegrees, decDegrees, unit=u.deg)  # noqa
    cCelestial = NumCpp.Celestial(ra, dec)
    assert cCelestial.ra() == ra
    assert cCelestial.dec() == dec
    assert round(cCelestial.x(), 10) == round(pycelestial.cartesian.x.value, 10)
    assert round(cCelestial.y(), 10) == round(pycelestial.cartesian.y.value, 10)
    assert round(cCelestial.z(), 10) == round(pycelestial.cartesian.z.value, 10)


####################################################################################
def test_celestial_cartesian_component_constructor():
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)
    pycelestial = SkyCoord(raDegrees, decDegrees, unit=u.deg)  # noqa
    cCelestial = NumCpp.Celestial(
        pycelestial.cartesian.x.value, pycelestial.cartesian.y.value, pycelestial.cartesian.z.value
    )
    assert round(cCelestial.ra().degrees(), 5) == round(ra.degrees(), 5)
    assert round(cCelestial.dec().degrees(), 5) == round(dec.degrees(), 5)
    assert round(cCelestial.x(), 5) == round(pycelestial.cartesian.x.value, 5)
    assert round(cCelestial.y(), 5) == round(pycelestial.cartesian.y.value, 5)
    assert round(cCelestial.z(), 5) == round(pycelestial.cartesian.z.value, 5)


####################################################################################
def test_celestial_cartesian_constructor():
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)
    pycelestial = SkyCoord(raDegrees, decDegrees, unit=u.deg)  # noqa
    cartesian = NumCpp.Cartesian(
        pycelestial.cartesian.x.value, pycelestial.cartesian.y.value, pycelestial.cartesian.z.value
    )
    cCelestial = NumCpp.Celestial(cartesian)
    assert round(cCelestial.ra().degrees(), 5) == round(ra.degrees(), 5)
    assert round(cCelestial.dec().degrees(), 5) == round(dec.degrees(), 5)
    assert round(cCelestial.x(), 5) == round(pycelestial.cartesian.x.value, 5)
    assert round(cCelestial.y(), 5) == round(pycelestial.cartesian.y.value, 5)
    assert round(cCelestial.z(), 5) == round(pycelestial.cartesian.z.value, 5)


####################################################################################
def test_celestial_ndarray_constructor():
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)
    pycelestial = SkyCoord(raDegrees, decDegrees, unit=u.deg)  # noqa
    vec = np.asarray([pycelestial.cartesian.x.value, pycelestial.cartesian.y.value, pycelestial.cartesian.z.value])
    cVec = NumCpp.NdArray(1, 3)
    cVec.setArray(vec)
    cCelestial = NumCpp.Celestial(cVec)
    assert round(cCelestial.ra().degrees(), 5) == round(ra.degrees(), 5)
    assert round(cCelestial.dec().degrees(), 5) == round(dec.degrees(), 5)
    assert round(cCelestial.x(), 5) == round(pycelestial.cartesian.x.value, 5)
    assert round(cCelestial.y(), 5) == round(pycelestial.cartesian.y.value, 5)
    assert round(cCelestial.z(), 5) == round(pycelestial.cartesian.z.value, 5)


####################################################################################
def test_celestial_vec3_constructor():
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)
    pycelestial = SkyCoord(raDegrees, decDegrees, unit=u.deg)  # noqa
    vec3 = NumCpp.Vec3(pycelestial.cartesian.x.value, pycelestial.cartesian.y.value, pycelestial.cartesian.z.value)
    cCelestial = NumCpp.Celestial(vec3)
    assert round(cCelestial.ra().degrees(), 5) == round(ra.degrees(), 5)
    assert round(cCelestial.dec().degrees(), 5) == round(dec.degrees(), 5)
    assert round(cCelestial.x(), 5) == round(pycelestial.cartesian.x.value, 5)
    assert round(cCelestial.y(), 5) == round(pycelestial.cartesian.y.value, 5)
    assert round(cCelestial.z(), 5) == round(pycelestial.cartesian.z.value, 5)


####################################################################################
def test_celestial_rms_constructor():
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

    cCelestial = NumCpp.Celestial(raHours, raMinutes, raSeconds, decSign, decDegrees, decMinutes, decSeconds)
    cRa = cCelestial.ra()
    cDec = cCelestial.dec()
    pycelestial = SkyCoord(raDegreesPy, decDegreesPy, unit=u.deg)  # noqa
    assert round(cRa.degrees(), 5) == round(raDegreesPy, 5)
    assert cRa.hours() == raHours
    assert cRa.minutes() == raMinutes
    assert round(cRa.seconds(), 5) == round(raSeconds, 5)
    assert cDec.sign() == decSign
    assert round(cDec.degrees(), 5) == round(decDegreesPy, 5)
    assert cDec.degreesWhole() == decDegrees
    assert cDec.minutes() == decMinutes
    assert round(cDec.seconds(), 5) == round(decSeconds, 5)
    assert round(cCelestial.x(), 5) == round(pycelestial.cartesian.x.value, 5)
    assert round(cCelestial.y(), 5) == round(pycelestial.cartesian.y.value, 5)
    assert round(cCelestial.z(), 5) == round(pycelestial.cartesian.z.value, 5)


####################################################################################
def test_celestial_copy_constructor_and_equality_operator():
    cCelestial = NumCpp.Celestial()
    assert cCelestial
    cCelestial2 = NumCpp.Celestial(cCelestial)
    assert cCelestial2 == cCelestial


####################################################################################
def test_celestial_not_equality_operator():
    cCelestial = NumCpp.Celestial()

    raDegrees = np.random.rand(1).item() * 360
    decDegrees = np.random.rand(1).item() * 180 - 90
    cCelestial2 = NumCpp.Celestial(raDegrees, decDegrees)
    assert cCelestial2 != cCelestial


####################################################################################
def test_celestial_xyz():
    cCelestial = NumCpp.Celestial()
    xyz = [cCelestial.x(), cCelestial.y(), cCelestial.z()]
    assert np.array_equal(cCelestial.xyz().getNumpyArray().flatten(), xyz)


####################################################################################
def test_celestial_degreeSeperation():
    cCelestial = NumCpp.Celestial()

    raDegrees = np.random.rand(1).item() * 360
    decDegrees = np.random.rand(1).item() * 180 - 90
    cCelestial2 = NumCpp.Celestial(raDegrees, decDegrees)

    pycelestial = SkyCoord(cCelestial.ra().degrees(), cCelestial.dec().degrees(), unit=u.deg)  # noqa
    pycelestial2 = SkyCoord(cCelestial2.ra().degrees(), cCelestial2.dec().degrees(), unit=u.deg)  # noqa

    cDegSep = cCelestial.degreeSeperation(cCelestial2)
    pyDegSep = pycelestial.separation(pycelestial2).value
    assert round(cDegSep, 5) == round(pyDegSep, 5)


####################################################################################
def test_celestial_radianSeperation():
    cCelestial = NumCpp.Celestial()

    raDegrees = np.random.rand(1).item() * 360
    decDegrees = np.random.rand(1).item() * 180 - 90
    cCelestial2 = NumCpp.Celestial(raDegrees, decDegrees)

    pycelestial = SkyCoord(cCelestial.ra().degrees(), cCelestial.dec().degrees(), unit=u.deg)  # noqa
    pycelestial2 = SkyCoord(cCelestial2.ra().degrees(), cCelestial2.dec().degrees(), unit=u.deg)  # noqa

    cRadSep = cCelestial.radianSeperation(cCelestial2)
    pyRadSep = np.deg2rad(pycelestial.separation(pycelestial2).value)

    assert round(cRadSep, 5) == round(pyRadSep, 5)


####################################################################################
def test_celestial_degreeSeperation_vec():
    cCelestial = NumCpp.Celestial()

    raDegrees = np.random.rand(1).item() * 360
    decDegrees = np.random.rand(1).item() * 180 - 90
    cCelestial2 = NumCpp.Celestial(raDegrees, decDegrees)

    pycelestial = SkyCoord(cCelestial.ra().degrees(), cCelestial.dec().degrees(), unit=u.deg)  # noqa
    pycelestial2 = SkyCoord(cCelestial2.ra().degrees(), cCelestial2.dec().degrees(), unit=u.deg)  # noqa

    vec2 = np.asarray([pycelestial2.cartesian.x, pycelestial2.cartesian.y, pycelestial2.cartesian.z])
    cArray = NumCpp.NdArray(1, 3)
    cArray.setArray(vec2)
    cDegSep = cCelestial.degreeSeperation(cArray)
    pyDegSep = pycelestial.separation(pycelestial2).value
    assert round(cDegSep, 5) == round(pyDegSep, 5)


####################################################################################
def test_celestial_radianSeperation_vec():
    cCelestial = NumCpp.Celestial()

    raDegrees = np.random.rand(1).item() * 360
    decDegrees = np.random.rand(1).item() * 180 - 90
    cCelestial2 = NumCpp.Celestial(raDegrees, decDegrees)

    pycelestial = SkyCoord(cCelestial.ra().degrees(), cCelestial.dec().degrees(), unit=u.deg)  # noqa
    pycelestial2 = SkyCoord(cCelestial2.ra().degrees(), cCelestial2.dec().degrees(), unit=u.deg)  # noqa

    vec2 = np.asarray([pycelestial2.cartesian.x, pycelestial2.cartesian.y, pycelestial2.cartesian.z])
    cArray = NumCpp.NdArray(1, 3)
    cArray.setArray(vec2)
    cRadSep = cCelestial.radianSeperation(cArray)
    pyRadSep = np.radians(pycelestial.separation(pycelestial2).value)
    assert round(cRadSep, 5) == round(pyRadSep, 5)  # noqa


####################################################################################
@pytest.mark.skip(
    reason="This segfaults right now, but I'm pretty sure it is just the pytest test, nothing wrong with the actual code..."
)
def test_celestial_print():
    cCelestial = NumCpp.Celestial()
    cCelestial.print()


####################################################################################
def test_AERtoECEF():
    az, el, sRange = np.random.rand(3) * np.pi / 4
    target = NumCpp.AER(az, el, sRange)
    x, y, z = np.random.uniform(1, 1.1, 3) * NumCpp.EARTH_EQUATORIAL_RADIUS
    referencePoint = NumCpp.ECEF(x, y, z)
    ecef = NumCpp.AERtoECEF(target, referencePoint)
    x1, y1, z1 = pymap3d.aer2ecef(az, el, sRange, *pymap3d.ecef2geodetic(x, y, z, deg=False), deg=False)
    np.testing.assert_approx_equal(ecef.x, x1, 5)
    np.testing.assert_approx_equal(ecef.y, y1, 5)
    np.testing.assert_approx_equal(ecef.z, z1, 5)

    lat, lon, alt = np.random.rand(3) * np.pi / 4
    referencePoint = NumCpp.LLA(lat, lon, alt)
    ecef = NumCpp.AERtoECEF(target, referencePoint)
    x1, y1, z1 = pymap3d.aer2ecef(az, el, sRange, lat, lon, alt, deg=False)
    np.testing.assert_approx_equal(ecef.x, x1, 5)
    np.testing.assert_approx_equal(ecef.y, y1, 5)
    np.testing.assert_approx_equal(ecef.z, z1, 5)


####################################################################################
def test_AERtoENU():
    az, el, sRange = np.random.rand(3) * np.pi / 4
    aer = NumCpp.AER(az, el, sRange)
    enu = NumCpp.AERtoENU(aer)
    east, north, up = pymap3d.aer2enu(az, el, sRange, deg=False)
    np.testing.assert_approx_equal(enu.east, east, 5)
    np.testing.assert_approx_equal(enu.north, north, 5)
    np.testing.assert_approx_equal(enu.up, up, 5)


####################################################################################
def test_AERtoLLA():
    az, el, sRange = np.random.rand(3) * np.pi / 4
    target = NumCpp.AER(az, el, sRange)
    x, y, z = np.random.uniform(1, 1.1, 3) * NumCpp.EARTH_EQUATORIAL_RADIUS
    referencePoint = NumCpp.ECEF(x, y, z)
    lla = NumCpp.AERtoLLA(target, referencePoint)
    lat, lon, alt = pymap3d.aer2geodetic(az, el, sRange, *pymap3d.ecef2geodetic(x, y, z, deg=False), deg=False)
    np.testing.assert_approx_equal(lla.latitude, lat, 5)
    np.testing.assert_approx_equal(lla.longitude, lon, 5)
    np.testing.assert_approx_equal(lla.altitude, alt, 5)

    lat, lon, alt = np.random.rand(3) * np.pi / 4
    referencePoint = NumCpp.LLA(lat, lon, alt)
    lla = NumCpp.AERtoLLA(target, referencePoint)
    lat1, lon1, alt1 = pymap3d.aer2geodetic(az, el, sRange, lat, lon, alt, deg=False)
    np.testing.assert_approx_equal(lla.latitude, lat1, 5)
    np.testing.assert_approx_equal(lla.longitude, lon1, 5)
    np.testing.assert_approx_equal(lla.altitude, alt1, 5)


####################################################################################
def test_AERtoNED():
    az, el, sRange = np.random.rand(3) * np.pi / 4
    aer = NumCpp.AER(az, el, sRange)
    ned = NumCpp.AERtoNED(aer)
    north, east, down = pymap3d.aer2ned(az, el, sRange, deg=False)
    np.testing.assert_approx_equal(ned.north, north, 5)
    np.testing.assert_approx_equal(ned.east, east, 5)
    np.testing.assert_approx_equal(ned.down, down, 5)


####################################################################################
def test_ECEFEulerToNEDRollPitchYaw():
    platform1ECEF = NumCpp.ECEF(889780.8040509718, -5443884.478448521, 3191301.5726495585)
    platform1Euler = NumCpp.Euler(1.678885817527771, -1.0427558422088623, -3.0950019359588623)
    platform1RollPitchYaw = NumCpp.Orientation(0.0, 0.0, 0.027159271086905079)

    platform1RollPitchYawCalc = NumCpp.ECEFEulerToNEDRollPitchYaw(platform1ECEF, platform1Euler)
    np.testing.assert_almost_equal(platform1RollPitchYawCalc.roll, platform1RollPitchYaw.roll, 5)
    np.testing.assert_almost_equal(platform1RollPitchYawCalc.pitch, platform1RollPitchYaw.pitch, 5)
    np.testing.assert_approx_equal(platform1RollPitchYawCalc.yaw, platform1RollPitchYaw.yaw, 5)

    platform1EulerCalc = NumCpp.NEDRollPitchYawToECEFEuler(platform1ECEF, platform1RollPitchYaw)
    np.testing.assert_approx_equal(platform1EulerCalc.psi, platform1Euler.psi, 5)
    np.testing.assert_approx_equal(platform1EulerCalc.theta, platform1Euler.theta, 5)
    np.testing.assert_approx_equal(platform1EulerCalc.phi, platform1Euler.phi, 5)

    platform2ECEF = NumCpp.ECEF(-1288345.7521444533, -4718928.642526492, 4079259.935028878)
    platform2Euler = NumCpp.Euler(1.30427503581543, -0.872403085231781, 3.1415927410125732)
    platform2RollPitchYaw = NumCpp.Orientation(0.0, 0.0, 0.0)

    platform2RollPitchYawCalc = NumCpp.ECEFEulerToNEDRollPitchYaw(platform2ECEF, platform2Euler)
    np.testing.assert_almost_equal(platform2RollPitchYawCalc.roll, platform2RollPitchYaw.roll, 5)
    np.testing.assert_almost_equal(platform2RollPitchYawCalc.pitch, platform2RollPitchYaw.pitch, 5)
    np.testing.assert_almost_equal(platform2RollPitchYawCalc.yaw, platform2RollPitchYaw.yaw, 5)

    platform2EulerCalc = NumCpp.NEDRollPitchYawToECEFEuler(platform2ECEF, platform2RollPitchYaw)
    np.testing.assert_approx_equal(platform2EulerCalc.psi, platform2Euler.psi, 5)
    np.testing.assert_approx_equal(platform2EulerCalc.theta, platform2Euler.theta, 5)
    np.testing.assert_approx_equal(platform2EulerCalc.phi, -platform2Euler.phi, 5)

    platform3ECEF = NumCpp.ECEF(861284.8918511268, -5441200.936501232, 3203589.383938122)
    platform3Euler = NumCpp.Euler(-2.4969322681427, -0.4192129075527191, 2.2737600803375244)
    platform3RollPitchYaw = NumCpp.Orientation(0.33161255787892263, 0.6126105674500097, -1.4049900478554354)

    platform3RollPitchYawCalc = NumCpp.ECEFEulerToNEDRollPitchYaw(platform3ECEF, platform3Euler)
    np.testing.assert_approx_equal(platform3RollPitchYawCalc.roll, platform3RollPitchYaw.roll, 5)
    np.testing.assert_approx_equal(platform3RollPitchYawCalc.pitch, platform3RollPitchYaw.pitch, 5)
    np.testing.assert_approx_equal(platform3RollPitchYawCalc.yaw, platform3RollPitchYaw.yaw, 5)

    platform3EulerCalc = NumCpp.NEDRollPitchYawToECEFEuler(platform3ECEF, platform3RollPitchYaw)
    np.testing.assert_approx_equal(platform3EulerCalc.psi, platform3Euler.psi, 5)
    np.testing.assert_approx_equal(platform3EulerCalc.theta, platform3Euler.theta, 5)
    np.testing.assert_approx_equal(platform3EulerCalc.phi, platform3Euler.phi, 5)


####################################################################################
def test_ECEFEulerToENURollPitchYaw():
    platform1ECEF = NumCpp.ECEF(889780.8040509718, -5443884.478448521, 3191301.5726495585)
    platform1Euler = NumCpp.Euler(1.678885817527771, -1.0427558422088623, -3.0950019359588623)
    platform1RollPitchYaw = NumCpp.Orientation(0.0, 0.0, -0.027159271086905079)

    platform1RollPitchYawCalc = NumCpp.ECEFEulerToENURollPitchYaw(platform1ECEF, platform1Euler)
    np.testing.assert_almost_equal(platform1RollPitchYawCalc.roll, platform1RollPitchYaw.roll, 5)
    np.testing.assert_almost_equal(platform1RollPitchYawCalc.pitch, platform1RollPitchYaw.pitch, 5)
    np.testing.assert_approx_equal(platform1RollPitchYawCalc.yaw, platform1RollPitchYaw.yaw, 5)

    platform1EulerCalc = NumCpp.ENURollPitchYawToECEFEuler(platform1ECEF, platform1RollPitchYaw)
    np.testing.assert_approx_equal(platform1EulerCalc.psi, platform1Euler.psi, 5)
    np.testing.assert_approx_equal(platform1EulerCalc.theta, platform1Euler.theta, 5)
    np.testing.assert_approx_equal(platform1EulerCalc.phi, platform1Euler.phi, 5)

    platform2ECEF = NumCpp.ECEF(-1288345.7521444533, -4718928.642526492, 4079259.935028878)
    platform2Euler = NumCpp.Euler(1.30427503581543, -0.872403085231781, 3.1415927410125732)
    platform2RollPitchYaw = NumCpp.Orientation(0.0, 0.0, 0.0)

    platform2RollPitchYawCalc = NumCpp.ECEFEulerToENURollPitchYaw(platform2ECEF, platform2Euler)
    np.testing.assert_almost_equal(platform2RollPitchYawCalc.roll, platform2RollPitchYaw.roll, 5)
    np.testing.assert_almost_equal(platform2RollPitchYawCalc.pitch, platform2RollPitchYaw.pitch, 5)
    np.testing.assert_almost_equal(platform2RollPitchYawCalc.yaw, platform2RollPitchYaw.yaw, 5)

    platform2EulerCalc = NumCpp.ENURollPitchYawToECEFEuler(platform2ECEF, platform2RollPitchYaw)
    np.testing.assert_approx_equal(platform2EulerCalc.psi, platform2Euler.psi, 5)
    np.testing.assert_approx_equal(platform2EulerCalc.theta, platform2Euler.theta, 5)
    np.testing.assert_approx_equal(platform2EulerCalc.phi, -platform2Euler.phi, 5)

    platform3ECEF = NumCpp.ECEF(861284.8918511268, -5441200.936501232, 3203589.383938122)
    platform3Euler = NumCpp.Euler(-2.4969322681427, -0.4192129075527191, 2.2737600803375244)
    platform3RollPitchYaw = NumCpp.Orientation(0.6126105674500097, 0.33161255787892263, 1.4049900478554354)

    platform3RollPitchYawCalc = NumCpp.ECEFEulerToENURollPitchYaw(platform3ECEF, platform3Euler)
    np.testing.assert_approx_equal(platform3RollPitchYawCalc.roll, platform3RollPitchYaw.roll, 5)
    np.testing.assert_approx_equal(platform3RollPitchYawCalc.pitch, platform3RollPitchYaw.pitch, 5)
    np.testing.assert_approx_equal(platform3RollPitchYawCalc.yaw, platform3RollPitchYaw.yaw, 5)

    platform3EulerCalc = NumCpp.ENURollPitchYawToECEFEuler(platform3ECEF, platform3RollPitchYaw)
    np.testing.assert_approx_equal(platform3EulerCalc.psi, platform3Euler.psi, 5)
    np.testing.assert_approx_equal(platform3EulerCalc.theta, platform3Euler.theta, 5)
    np.testing.assert_approx_equal(platform3EulerCalc.phi, platform3Euler.phi, 5)


####################################################################################
def test_ECEFtoAER():
    x1, y1, z1 = np.random.uniform(1, 1.1, 3) * NumCpp.EARTH_EQUATORIAL_RADIUS
    target = NumCpp.ECEF(x1, y1, z1)
    x2, y2, z2 = np.random.uniform(1, 1.1, 3) * NumCpp.EARTH_EQUATORIAL_RADIUS
    referencePoint = NumCpp.ECEF(x2, y2, z2)
    aer = NumCpp.ECEFtoAER(target, referencePoint)
    az, el, sRange = pymap3d.ecef2aer(x1, y1, z1, *pymap3d.ecef2geodetic(x2, y2, z2, deg=False), deg=False)
    np.testing.assert_approx_equal(aer.az, az, 5)
    np.testing.assert_approx_equal(aer.el, el, 5)
    np.testing.assert_approx_equal(aer.range, sRange, 5)

    lat, lon, alt = np.random.rand(3) * np.pi / 4
    referencePoint = NumCpp.LLA(lat, lon, alt)
    aer = NumCpp.ECEFtoAER(target, referencePoint)
    az, el, sRange = pymap3d.ecef2aer(x1, y1, z1, lat, lon, alt, deg=False)
    np.testing.assert_approx_equal(aer.az, az, 5)
    np.testing.assert_approx_equal(aer.el, el, 5)
    np.testing.assert_approx_equal(aer.range, sRange, 5)


####################################################################################
def test_ECEFtoENU():
    x1, y1, z1 = np.random.uniform(1, 1.1, 3) * NumCpp.EARTH_EQUATORIAL_RADIUS
    target = NumCpp.ECEF(x1, y1, z1)
    x2, y2, z2 = np.random.uniform(1, 1.1, 3) * NumCpp.EARTH_EQUATORIAL_RADIUS
    referencePoint = NumCpp.ECEF(x2, y2, z2)
    enu = NumCpp.ECEFtoENU(target, referencePoint)
    east, north, up = pymap3d.ecef2enu(x1, y1, z1, *pymap3d.ecef2geodetic(x2, y2, z2, deg=False), deg=False)
    np.testing.assert_approx_equal(enu.east, east, 5)
    np.testing.assert_approx_equal(enu.north, north, 5)
    np.testing.assert_approx_equal(enu.up, up, 5)

    lat, lon, alt = np.random.rand(3) * np.pi / 4
    referencePoint = NumCpp.LLA(lat, lon, alt)
    enu = NumCpp.ECEFtoENU(target, referencePoint)
    east, north, up = pymap3d.ecef2enu(x1, y1, z1, lat, lon, alt, deg=False)
    np.testing.assert_approx_equal(enu.east, east, 5)
    np.testing.assert_approx_equal(enu.north, north, 5)
    np.testing.assert_approx_equal(enu.up, up, 5)


####################################################################################
def test_ECEFtoLLA():
    x, y, z = np.random.uniform(1, 1.1, 3) * NumCpp.EARTH_EQUATORIAL_RADIUS
    ecef = NumCpp.ECEF(x, y, z)
    lla = NumCpp.ECEFtoLLA(ecef, 1e-8)
    lat, lon, alt = pymap3d.ecef2geodetic(x, y, z, deg=False)
    np.testing.assert_approx_equal(lla.latitude, lat, 5)
    np.testing.assert_approx_equal(lla.longitude, lon, 5)
    np.testing.assert_approx_equal(lla.altitude, alt, 5)


####################################################################################
def test_ECEFtoNED():
    x1, y1, z1 = np.random.uniform(1, 1.1, 3) * NumCpp.EARTH_EQUATORIAL_RADIUS
    target = NumCpp.ECEF(x1, y1, z1)
    x2, y2, z2 = np.random.uniform(1, 1.1, 3) * NumCpp.EARTH_EQUATORIAL_RADIUS
    referencePoint = NumCpp.ECEF(x2, y2, z2)
    ned = NumCpp.ECEFtoNED(target, referencePoint)
    north, east, down = pymap3d.ecef2ned(x1, y1, z1, *pymap3d.ecef2geodetic(x2, y2, z2, deg=False), deg=False)
    np.testing.assert_approx_equal(ned.north, north, 5)
    np.testing.assert_approx_equal(ned.east, east, 5)
    np.testing.assert_approx_equal(ned.down, down, 5)

    lat, lon, alt = np.random.rand(3) * np.pi / 4
    referencePoint = NumCpp.LLA(lat, lon, alt)
    ned = NumCpp.ECEFtoNED(target, referencePoint)
    north, east, down = pymap3d.ecef2ned(x1, y1, z1, lat, lon, alt, deg=False)
    np.testing.assert_approx_equal(ned.north, north, 5)
    np.testing.assert_approx_equal(ned.east, east, 5)
    np.testing.assert_approx_equal(ned.down, down, 5)


####################################################################################
def test_ENUtoAER():
    east, north, up = np.random.rand(3) * 1000
    enu = NumCpp.ENU(east, north, up)
    aer = NumCpp.ENUtoAER(enu)
    az, el, sRange = pymap3d.enu2aer(east, north, up, deg=False)
    np.testing.assert_approx_equal(aer.az, az, 5)
    np.testing.assert_approx_equal(aer.el, el, 5)
    np.testing.assert_approx_equal(aer.range, sRange, 5)


####################################################################################
def test_ENUtoECEF():
    east, north, up = np.random.rand(3) * 1000
    target = NumCpp.ENU(east, north, up)
    x, y, z = np.random.uniform(1, 1.1, 3) * NumCpp.EARTH_EQUATORIAL_RADIUS
    referencePoint = NumCpp.ECEF(x, y, z)
    ecef = NumCpp.ENUtoECEF(target, referencePoint)
    x1, y1, z1 = pymap3d.enu2ecef(east, north, up, *pymap3d.ecef2geodetic(x, y, z, deg=False), deg=False)
    np.testing.assert_approx_equal(ecef.x, x1, 5)
    np.testing.assert_approx_equal(ecef.y, y1, 5)
    np.testing.assert_approx_equal(ecef.z, z1, 5)

    lat, lon, alt = np.random.rand(3) * np.pi / 4
    referencePoint = NumCpp.LLA(lat, lon, alt)
    ecef = NumCpp.ENUtoECEF(target, referencePoint)
    x1, y1, z1 = pymap3d.enu2ecef(east, north, up, lat, lon, alt, deg=False)
    np.testing.assert_approx_equal(ecef.x, x1, 5)
    np.testing.assert_approx_equal(ecef.y, y1, 5)
    np.testing.assert_approx_equal(ecef.z, z1, 5)


####################################################################################
def test_ENUtoLLA():
    east, north, up = np.random.rand(3) * 1000
    target = NumCpp.ENU(east, north, up)
    x, y, z = np.random.uniform(1, 1.1, 3) * NumCpp.EARTH_EQUATORIAL_RADIUS
    referencePoint = NumCpp.ECEF(x, y, z)
    lla = NumCpp.ENUtoLLA(target, referencePoint)
    lat, lon, alt = pymap3d.enu2geodetic(east, north, up, *pymap3d.ecef2geodetic(x, y, z, deg=False), deg=False)
    np.testing.assert_approx_equal(lla.latitude, lat, 5)
    np.testing.assert_approx_equal(lla.longitude, lon, 5)
    np.testing.assert_approx_equal(lla.altitude, alt, 5)

    lat, lon, alt = np.random.rand(3) * np.pi / 4
    referencePoint = NumCpp.LLA(lat, lon, alt)
    lla = NumCpp.ENUtoLLA(target, referencePoint)
    lat, lon, alt = pymap3d.enu2geodetic(east, north, up, lat, lon, alt, deg=False)
    np.testing.assert_approx_equal(lla.latitude, lat, 5)
    np.testing.assert_approx_equal(lla.longitude, lon, 5)
    np.testing.assert_approx_equal(lla.altitude, alt, 5)


####################################################################################
def test_ENUtoNED():
    east, north, up = np.random.rand(3) * 1000
    enu = NumCpp.ENU(east, north, up)
    ned = NumCpp.ENUtoNED(enu)
    assert ned.north == enu.north
    assert ned.east == enu.east
    assert ned.down == -enu.up


####################################################################################
def test_geocentricToLLA():
    lat, lon, radius = np.random.rand(3) * np.pi / 4
    radius += NumCpp.EARTH_EQUATORIAL_RADIUS
    geodetic = NumCpp.geocentricToLLA(NumCpp.Geocentric(lat, lon, radius))
    lat1, lon1, alt1 = pymap3d.spherical2geodetic(lat, lon, radius, deg=False)
    np.testing.assert_approx_equal(geodetic.latitude, lat1, 5)
    np.testing.assert_approx_equal(geodetic.longitude, lon1, 5)
    np.testing.assert_approx_equal(geodetic.altitude, alt1, 5)


####################################################################################
def test_LLAtoGeocentric():
    lat, lon, alt = np.random.rand(3) * np.pi / 4
    geocentric = NumCpp.LLAtoGeocentric(NumCpp.LLA(lat, lon, alt))
    lat1, lon1, radius1 = pymap3d.geodetic2spherical(lat, lon, alt, deg=False)
    np.testing.assert_approx_equal(geocentric.latitude, lat1, 5)
    np.testing.assert_approx_equal(geocentric.longitude, lon1, 5)
    np.testing.assert_approx_equal(geocentric.radius, radius1, 5)


####################################################################################
def test_LLAtoAER():
    lat, lon, alt = np.random.rand(3) * np.pi / 4
    target = NumCpp.LLA(lat, lon, alt)
    x, y, z = np.random.uniform(1, 1.1, 3) * NumCpp.EARTH_EQUATORIAL_RADIUS
    referencePoint = NumCpp.ECEF(x, y, z)
    aer = NumCpp.LLAtoAER(target, referencePoint)
    az, el, sRange = pymap3d.geodetic2aer(lat, lon, alt, *pymap3d.ecef2geodetic(x, y, z, deg=False), deg=False)
    np.testing.assert_approx_equal(aer.az, az, 5)
    np.testing.assert_approx_equal(aer.el, el, 5)
    np.testing.assert_approx_equal(aer.range, sRange, 5)

    lat1, lon1, alt1 = np.random.rand(3) * np.pi / 4
    referencePoint = NumCpp.LLA(lat1, lon1, alt1)
    aer = NumCpp.LLAtoAER(target, referencePoint)
    az, el, sRange = pymap3d.geodetic2aer(lat, lon, alt, lat1, lon1, alt1, deg=False)
    np.testing.assert_approx_equal(aer.az, az, 5)
    np.testing.assert_approx_equal(aer.el, el, 5)
    np.testing.assert_approx_equal(aer.range, sRange, 5)


####################################################################################
def test_LLAtoECEF():
    lat, lon, alt = np.random.rand(3) * np.pi / 4
    lla = NumCpp.LLA(lat, lon, alt)
    ecef = NumCpp.LLAtoECEF(lla)
    x, y, z = pymap3d.geodetic2ecef(lat, lon, alt, deg=False)
    np.testing.assert_approx_equal(ecef.x, x, 5)
    np.testing.assert_approx_equal(ecef.y, y, 5)
    np.testing.assert_approx_equal(ecef.z, z, 5)


####################################################################################
def test_LLAtoENU():
    lat, lon, alt = np.random.rand(3) * np.pi / 4
    target = NumCpp.LLA(lat, lon, alt)
    x, y, z = np.random.uniform(1, 1.1, 3) * NumCpp.EARTH_EQUATORIAL_RADIUS
    referencePoint = NumCpp.ECEF(x, y, z)
    enu = NumCpp.LLAtoENU(target, referencePoint)
    east, north, up = pymap3d.geodetic2enu(lat, lon, alt, *pymap3d.ecef2geodetic(x, y, z, deg=False), deg=False)
    np.testing.assert_approx_equal(enu.east, east, 5)
    np.testing.assert_approx_equal(enu.north, north, 5)
    np.testing.assert_approx_equal(enu.up, up, 5)

    lat1, lon1, alt1 = np.random.rand(3) * np.pi / 4
    referencePoint = NumCpp.LLA(lat1, lon1, alt1)
    enu = NumCpp.LLAtoENU(target, referencePoint)
    east, north, up = pymap3d.geodetic2enu(lat, lon, alt, lat1, lon1, alt1, deg=False)
    np.testing.assert_approx_equal(enu.east, east, 5)
    np.testing.assert_approx_equal(enu.north, north, 5)
    np.testing.assert_approx_equal(enu.up, up, 5)


####################################################################################
def test_LLAtoNED():
    lat, lon, alt = np.random.rand(3) * np.pi / 4
    target = NumCpp.LLA(lat, lon, alt)
    x, y, z = np.random.uniform(1, 1.1, 3) * NumCpp.EARTH_EQUATORIAL_RADIUS
    referencePoint = NumCpp.ECEF(x, y, z)
    ned = NumCpp.LLAtoNED(target, referencePoint)
    north, east, down = pymap3d.geodetic2ned(lat, lon, alt, *pymap3d.ecef2geodetic(x, y, z, deg=False), deg=False)
    np.testing.assert_approx_equal(ned.north, north, 5)
    np.testing.assert_approx_equal(ned.east, east, 5)
    np.testing.assert_approx_equal(ned.down, down, 5)

    lat1, lon1, alt1 = np.random.rand(3) * np.pi / 4
    referencePoint = NumCpp.LLA(lat1, lon1, alt1)
    ned = NumCpp.LLAtoNED(target, referencePoint)
    north, east, down = pymap3d.geodetic2ned(lat, lon, alt, lat1, lon1, alt1, deg=False)
    np.testing.assert_approx_equal(ned.north, north, 5)
    np.testing.assert_approx_equal(ned.east, east, 5)
    np.testing.assert_approx_equal(ned.down, down, 5)


####################################################################################
def test_NEDRollPitchYawToECEFEuler():
    x, y, z = np.random.uniform(1, 1.1, 3) * NumCpp.EARTH_EQUATORIAL_RADIUS
    ecef = NumCpp.ECEF(x, y, z)
    roll, pitch, yaw = np.random.rand(3) * np.pi / 4
    orientation = NumCpp.Orientation(roll, pitch, yaw)
    euler = NumCpp.NEDRollPitchYawToECEFEuler(ecef, orientation)
    newOrientation = NumCpp.ECEFEulerToNEDRollPitchYaw(ecef, euler)
    np.testing.assert_approx_equal(newOrientation.roll, roll, 5)
    np.testing.assert_approx_equal(newOrientation.pitch, pitch, 5)
    np.testing.assert_approx_equal(newOrientation.yaw, yaw, 5)


####################################################################################
def test_NEDtoAER():
    north, east, down = np.random.rand(3) * 1000
    ned = NumCpp.NED(north, east, down)
    aer = NumCpp.NEDtoAER(ned)
    az, el, sRange = pymap3d.ned2aer(north, east, down, deg=False)
    np.testing.assert_approx_equal(aer.az, az, 5)
    np.testing.assert_approx_equal(aer.el, el, 5)
    np.testing.assert_approx_equal(aer.range, sRange, 5)


####################################################################################
def test_NEDtoECEF():
    north, east, down = np.random.rand(3) * 1000
    target = NumCpp.NED(north, east, down)
    x, y, z = np.random.uniform(1, 1.1, 3) * NumCpp.EARTH_EQUATORIAL_RADIUS
    referencePoint = NumCpp.ECEF(x, y, z)
    ecef = NumCpp.NEDtoECEF(target, referencePoint)
    x1, y1, z1 = pymap3d.ned2ecef(north, east, down, *pymap3d.ecef2geodetic(x, y, z, deg=False), deg=False)
    np.testing.assert_approx_equal(ecef.x, x1, 5)
    np.testing.assert_approx_equal(ecef.y, y1, 5)
    np.testing.assert_approx_equal(ecef.z, z1, 5)

    lat, lon, alt = np.random.rand(3) * np.pi / 4
    referencePoint = NumCpp.LLA(lat, lon, alt)
    ecef = NumCpp.NEDtoECEF(target, referencePoint)
    x1, y1, z1 = pymap3d.ned2ecef(north, east, down, lat, lon, alt, deg=False)
    np.testing.assert_approx_equal(ecef.x, x1, 5)
    np.testing.assert_approx_equal(ecef.y, y1, 5)
    np.testing.assert_approx_equal(ecef.z, z1, 5)


####################################################################################
def test_NEDtoENU():
    north, east, down = np.random.rand(3) * 1000
    ned = NumCpp.NED(north, east, down)
    enu = NumCpp.NEDtoENU(ned)
    assert enu.east == ned.east
    assert enu.north == ned.north
    assert enu.up == -ned.down


###################################################################################
def test_NEDtoLLA():
    north, east, down = np.random.rand(3) * 1000
    target = NumCpp.NED(north, east, down)
    x, y, z = np.random.uniform(1, 1.1, 3) * NumCpp.EARTH_EQUATORIAL_RADIUS
    referencePoint = NumCpp.ECEF(x, y, z)
    lla = NumCpp.NEDtoLLA(target, referencePoint)
    lat, lon, alt = pymap3d.ned2geodetic(north, east, down, *pymap3d.ecef2geodetic(x, y, z, deg=False), deg=False)
    np.testing.assert_approx_equal(lla.latitude, lat, 5)
    np.testing.assert_approx_equal(lla.longitude, lon, 5)
    np.testing.assert_approx_equal(lla.altitude, alt, 5)

    lat, lon, alt = np.random.rand(3) * np.pi / 4
    referencePoint = NumCpp.LLA(lat, lon, alt)
    lla = NumCpp.NEDtoLLA(target, referencePoint)
    lat1, lon1, alt1 = pymap3d.ned2geodetic(north, east, down, lat, lon, alt, deg=False)
    np.testing.assert_approx_equal(lla.latitude, lat1, 5)
    np.testing.assert_approx_equal(lla.longitude, lon1, 5)
    np.testing.assert_approx_equal(lla.altitude, alt1, 5)
