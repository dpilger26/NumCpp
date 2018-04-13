import numpy as np
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
from termcolor import colored
import sys
sys.path.append(r'../build/x64/Release')
import NumC

####################################################################################
def doTest():
    print(colored('Testing Coordinates Module', 'magenta'))

    print(colored('Testing Ra', 'magenta'))
    print(colored('Testing Default Constructor', 'cyan'))
    ra = NumC.RaDouble()
    print(colored('\tPASS', 'green'))

    print(colored('Testing Degree Constructor', 'cyan'))
    randDegrees = np.random.rand(1).item() * 360
    ra = NumC.RaDouble(randDegrees)
    raPy = Longitude(randDegrees, unit=u.deg)
    if (round(ra.degrees(), 9) == round(randDegrees, 9) and
        ra.hours() == raPy.hms.h and
        ra.minutes() == raPy.hms.m and
        round(ra.seconds(), 9) == round(raPy.hms.s, 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing hms Constructor', 'cyan'))
    hours = np.random.randint(0, 24, [1,], dtype=np.uint8).item()
    minutes = np.random.randint(0, 60, [1, ], dtype=np.uint8).item()
    seconds = np.random.rand(1).astype(np.float32).item() * 60
    ra = NumC.RaDouble(hours, minutes, seconds)
    degreesPy = (hours + minutes / 60 + seconds / 3600) * 15
    if (round(ra.degrees(), 9) == round(degreesPy, 9) and
        ra.hours() == hours and
        ra.minutes() == minutes and
        round(ra.seconds(), 9) == round(seconds, 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing astype', 'cyan'))
    raF = ra.asFloat()
    if (round(ra.degrees(), 4) == round(raF.degrees(), 4) and
        ra.hours() == raF.hours() and
        ra.minutes() == raF.minutes() and
        round(ra.seconds(), 4) == round(raF.seconds(), 4)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing equality operator', 'cyan'))
    ra2 = NumC.RaDouble(ra)
    if ra == ra2:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing not equality operator', 'cyan'))
    randDegrees = np.random.rand(1).item() * 360
    ra2 = NumC.RaDouble(randDegrees)
    if ra != ra2:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing print', 'cyan'))
    ra.print()
    print(colored('\tPASS', 'green'))

    print(colored('Testing Dec', 'magenta'))
    print(colored('Testing Default Constructor', 'cyan'))
    dec = NumC.DecDouble()
    print(colored('\tPASS', 'green'))

    print(colored('Testing Degree Constructor', 'cyan'))
    randDegrees = np.random.rand(1).item() * 180 - 90
    dec = NumC.DecDouble(randDegrees)
    decPy = Latitude(randDegrees, unit=u.deg)
    if (round(dec.degrees(), 10) == round(randDegrees, 10) and
        dec.degreesWhole() == decPy.dms.d and
        dec.minutes() == decPy.dms.m and
        round(dec.seconds(), 10) == round(decPy.dms.s, 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing dms Constructor', 'cyan'))
    degrees = np.random.randint(-90, 91, [1, ], dtype=np.uint8).item()
    minutes = np.random.randint(0, 60, [1, ], dtype=np.uint8).item()
    seconds = np.random.rand(1).astype(np.float32).item() * 60
    dec = NumC.DecDouble(hours, minutes, seconds)
    degreesPy = degrees + minutes / 60 + seconds / 3600
    if (round(dec.degrees(), 9) == round(degreesPy, 9) and
        dec.degreesWhole() == degrees and
        dec.minutes() == minutes and
        round(dec.seconds(), 9) == round(seconds, 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    return

    print(colored('Testing astype', 'cyan'))
    raF = ra.asFloat()
    if (round(ra.degrees(), 4) == round(raF.degrees(), 4) and
        ra.hours() == raF.hours() and
        ra.minutes() == raF.minutes() and
        round(ra.seconds(), 4) == round(raF.seconds(), 4)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing equality operator', 'cyan'))
    ra2 = NumC.RaDouble(ra)
    if ra == ra2:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing not equality operator', 'cyan'))
    randDegrees = np.random.rand(1).item() * 360
    ra2 = NumC.RaDouble(randDegrees)
    if ra != ra2:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing print', 'cyan'))
    ra.print()
    print(colored('\tPASS', 'green'))

####################################################################################
if __name__ == '__main__':
    doTest()