import numpy as np
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
from termcolor import colored
import sys
if sys.platform == 'linux':
    sys.path.append(r'../lib')
else:
    sys.path.append(r'../build/x64/Release')
import NumCpp

####################################################################################
def doTest():
    print(colored('Testing Coordinates Module', 'magenta'))

    print(colored('Testing Ra', 'magenta'))
    print(colored('Testing Default Constructor', 'cyan'))
    ra = NumCpp.Ra()
    print(colored('\tPASS', 'green'))

    print(colored('Testing Degree Constructor', 'cyan'))
    randDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(randDegrees)
    raPy = Longitude(randDegrees, unit=u.deg)
    if (round(ra.degrees(), 9) == round(randDegrees, 9) and
        ra.hours() == raPy.hms.h and
        ra.minutes() == raPy.hms.m and
        round(ra.seconds(), 9) == round(raPy.hms.s, 9) and
        round(ra.radians(), 9) == round(np.deg2rad(randDegrees), 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing hms Constructor', 'cyan'))
    hours = np.random.randint(0, 24, [1,], dtype=np.uint8).item()
    minutes = np.random.randint(0, 60, [1, ], dtype=np.uint8).item()
    seconds = np.random.rand(1).astype(np.double).item() * 60
    ra = NumCpp.Ra(hours, minutes, seconds)
    degreesPy = (hours + minutes / 60 + seconds / 3600) * 15
    if (round(ra.degrees(), 9) == round(degreesPy, 9) and
        ra.hours() == hours and
        ra.minutes() == minutes and
        round(ra.seconds(), 9) == round(seconds, 9) and
        round(ra.radians(), 9) == round(np.deg2rad(degreesPy), 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing equality operator', 'cyan'))
    ra2 = NumCpp.Ra(ra)
    if ra == ra2:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing not equality operator', 'cyan'))
    randDegrees = np.random.rand(1).item() * 360
    ra2 = NumCpp.Ra(randDegrees)
    if ra != ra2:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing print', 'cyan'))
    ra.print()
    print(colored('\tPASS', 'green'))

    print(colored('Testing Dec', 'magenta'))
    print(colored('Testing Default Constructor', 'cyan'))
    dec = NumCpp.Dec()
    print(colored('\tPASS', 'green'))

    print(colored('Testing Degree Constructor', 'cyan'))
    randDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(randDegrees)
    decPy = Latitude(randDegrees, unit=u.deg)
    sign = NumCpp.Sign.NEGATIVE if randDegrees < 0 else NumCpp.Sign.POSITIVE
    if (round(dec.degrees(), 8) == round(randDegrees, 8) and
        dec.sign() == sign and
        dec.degreesWhole() == abs(decPy.dms.d) and
        dec.minutes() == abs(decPy.dms.m) and
        round(dec.seconds(), 8) == round(abs(decPy.dms.s), 8) and
        round(dec.radians(), 8) == round(np.deg2rad(randDegrees), 8)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing dms Constructor', 'cyan'))
    sign = NumCpp.Sign.POSITIVE if np.random.randint(-1, 1) == 0 else NumCpp.Sign.NEGATIVE
    degrees = np.random.randint(0, 91, [1, ], dtype=np.uint8).item()
    minutes = np.random.randint(0, 60, [1, ], dtype=np.uint8).item()
    seconds = np.random.rand(1).astype(np.double).item() * 60
    dec = NumCpp.Dec(sign, degrees, minutes, seconds)
    degreesPy = degrees + minutes / 60 + seconds / 3600
    if sign == NumCpp.Sign.NEGATIVE:
        degreesPy *= -1
    if (dec.sign() == sign and
        round(dec.degrees(), 9) == round(degreesPy, 9) and
        dec.degreesWhole() == degrees and
        dec.minutes() == minutes and
        round(dec.seconds(), 9) == round(seconds, 9) and
        round(dec.radians(), 8) == round(np.deg2rad(degreesPy), 8)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing equality operator', 'cyan'))
    dec2 = NumCpp.Dec(dec)
    if dec == dec2:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing not equality operator', 'cyan'))
    randDegrees = np.random.rand(1).item() * 180 - 90
    dec2 = NumCpp.Dec(randDegrees)
    if dec != dec2:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing print', 'cyan'))
    dec.print()
    print(colored('\tPASS', 'green'))

    print(colored('Testing Coordinate', 'magenta'))
    print(colored('Testing Default Constructor', 'cyan'))
    coord = NumCpp.Coordinate()
    print(colored('\tPASS', 'green'))

    print(colored('Testing Degree Constructor', 'cyan'))
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)

    pyCoord = SkyCoord(raDegrees, decDegrees, unit=u.deg)
    cCoord = NumCpp.Coordinate(raDegrees, decDegrees)
    if (cCoord.ra() == ra and
        cCoord.dec() == dec and
        round(cCoord.x(), 10) == round(pyCoord.cartesian.x.value, 10) and
        round(cCoord.y(), 10) == round(pyCoord.cartesian.y.value, 10) and
        round(cCoord.z(), 10) == round(pyCoord.cartesian.z.value, 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Ra/Dec Constructor', 'cyan'))
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)

    pyCoord = SkyCoord(raDegrees, decDegrees, unit=u.deg)
    cCoord = NumCpp.Coordinate(ra, dec)
    if (cCoord.ra() == ra and
        cCoord.dec() == dec and
        round(cCoord.x(), 10) == round(pyCoord.cartesian.x.value, 10) and
        round(cCoord.y(), 10) == round(pyCoord.cartesian.y.value, 10) and
        round(cCoord.z(), 10) == round(pyCoord.cartesian.z.value, 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing x/y/z Constructor', 'cyan'))
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)
    pyCoord = SkyCoord(raDegrees, decDegrees, unit=u.deg)
    cCoord = NumCpp.Coordinate(pyCoord.cartesian.x.value, pyCoord.cartesian.y.value, pyCoord.cartesian.z.value)
    if (round(cCoord.ra().degrees(), 9) == round(ra.degrees(), 9) and
        round(cCoord.dec().degrees(), 9) == round(dec.degrees(), 9) and
        round(cCoord.x(), 9) == round(pyCoord.cartesian.x.value, 9) and
        round(cCoord.y(), 9) == round(pyCoord.cartesian.y.value, 9) and
        round(cCoord.z(), 9) == round(pyCoord.cartesian.z.value, 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing NdArray Constructor', 'cyan'))
    raDegrees = np.random.rand(1).item() * 360
    ra = NumCpp.Ra(raDegrees)
    decDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumCpp.Dec(decDegrees)
    pyCoord = SkyCoord(raDegrees, decDegrees, unit=u.deg)
    vec = np.asarray([pyCoord.cartesian.x.value, pyCoord.cartesian.y.value, pyCoord.cartesian.z.value])
    cVec = NumCpp.NdArray(1, 3)
    cVec.setArray(vec)
    cCoord = NumCpp.Coordinate(cVec)
    if (round(cCoord.ra().degrees(), 9) == round(ra.degrees(), 9) and
        round(cCoord.dec().degrees(), 9) == round(dec.degrees(), 9) and
        round(cCoord.x(), 9) == round(pyCoord.cartesian.x.value, 9) and
        round(cCoord.y(), 9) == round(pyCoord.cartesian.y.value, 9) and
        round(cCoord.z(), 9) == round(pyCoord.cartesian.z.value, 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing hms/dms Constructor', 'cyan'))
    raHours = np.random.randint(0, 24, [1, ], dtype=np.uint8).item()
    raMinutes = np.random.randint(0, 60, [1, ], dtype=np.uint8).item()
    raSeconds = np.random.rand(1).astype(np.double).item() * 60
    raDegreesPy = (raHours + raMinutes / 60 + raSeconds / 3600) * 15

    decSign = NumCpp.Sign.POSITIVE if np.random.randint(-1, 1) == 0 else NumCpp.Sign.NEGATIVE
    decDegrees = np.random.randint(0, 90, [1, ], dtype=np.uint8).item()
    decMinutes = np.random.randint(0, 60, [1, ], dtype=np.uint8).item()
    decSeconds = np.random.rand(1).astype(np.double).item() * 60
    decDegreesPy = decDegrees + decMinutes / 60 + decSeconds / 3600
    if decSign == NumCpp.Sign.NEGATIVE:
        decDegreesPy *= -1

    cCoord = NumCpp.Coordinate(raHours, raMinutes, raSeconds, decSign, decDegrees, decMinutes, decSeconds)
    cRa = cCoord.ra()
    cDec = cCoord.dec()
    pyCoord = SkyCoord(raDegreesPy, decDegreesPy, unit=u.deg)
    if (round(cRa.degrees(), 9) == round(raDegreesPy, 9) and
        cRa.hours() == raHours and
        cRa.minutes() == raMinutes and
        round(cRa.seconds(), 9) == round(raSeconds, 9) and
        cDec.sign() == decSign and
        round(cDec.degrees(), 9) == round(decDegreesPy, 9) and
        cDec.degreesWhole() == decDegrees and
        cDec.minutes() == decMinutes and
        round(cDec.seconds(), 9) == round(decSeconds, 9) and
        round(cCoord.x(), 9) == round(pyCoord.cartesian.x.value, 9) and
        round(cCoord.y(), 9) == round(pyCoord.cartesian.y.value, 9) and
        round(cCoord.z(), 9) == round(pyCoord.cartesian.z.value, 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing equality operator', 'cyan'))
    cCoord2 = NumCpp.Coordinate(cCoord)
    if cCoord2 == cCoord:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing not equality operator', 'cyan'))
    raDegrees = np.random.rand(1).item() * 360
    decDegrees = np.random.rand(1).item() * 180 - 90
    cCoord2 = NumCpp.Coordinate(raDegrees, decDegrees)
    if cCoord2 != cCoord:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing xyz', 'cyan'))
    xyz = [cCoord.x(), cCoord.y(), cCoord.z()]
    if np.array_equal(cCoord.xyz().getNumpyArray().flatten(), xyz):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing degreeSeperation Coordinate', 'cyan'))
    pyCoord2 = SkyCoord(cCoord2.ra().degrees(), cCoord2.dec().degrees(), unit=u.deg)
    cDegSep = cCoord.degreeSeperation(cCoord2)
    pyDegSep = pyCoord.separation(pyCoord2).value
    if round(cDegSep, 9) == round(pyDegSep, 9):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing radianSeperation Coordinate', 'cyan'))
    cRadSep = cCoord.radianSeperation(cCoord2)
    pyRadSep = np.deg2rad(pyDegSep)
    if round(cRadSep, 9) == round(pyRadSep, 9):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing degreeSeperation Vector', 'cyan'))
    vec2 = np.asarray([pyCoord2.cartesian.x, pyCoord2.cartesian.y, pyCoord2.cartesian.z])
    cArray = NumCpp.NdArray(1, 3)
    cArray.setArray(vec2)
    cDegSep = cCoord.degreeSeperation(cArray)
    if round(cDegSep, 9) == round(pyDegSep, 9):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing radianSeperation Vector', 'cyan'))
    cArray = NumCpp.NdArray(1, 3)
    cArray.setArray(vec2)
    cDegSep = cCoord.radianSeperation(cArray)
    if round(cDegSep, 9) == round(pyRadSep, 9):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing print', 'cyan'))
    cCoord.print()
    print(colored('\tPASS', 'green'))

####################################################################################
if __name__ == '__main__':
    doTest()
