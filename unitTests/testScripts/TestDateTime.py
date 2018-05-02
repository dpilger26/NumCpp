import numpy as np
from termcolor import colored
import datetime
import time
import sys
sys.path.append(r'../build/x64/Release')
import NumC

####################################################################################
def doTest():
    print(colored('Testing DateTime Module', 'magenta'))

    print(colored('Testing Default Constructor', 'cyan'))
    d = NumC.DateTime()
    if d.datetime() == 0:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Value Constructor', 'cyan'))
    year = np.random.randint(1971, 2070, [1,]).item()
    month = np.random.randint(1, 13, [1, ]).item()
    day = np.random.randint(1, 32, [1, ]).item()
    hour = np.random.randint(0, 24, [1, ]).item()
    minute = np.random.randint(0, 60, [1, ]).item()
    second = np.random.randint(0, 60, [1, ]).item()
    d = NumC.DateTime(year, month, day, hour, minute, second)
    if (d.year() == year and
            d.month() == month and
            d.day() == day and
            d.hour() == hour and
            d.minute() == minute and
            d.second() == second):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing now', 'cyan'))
    now = NumC.DateTime.now()
    pyNow = datetime.datetime.now()
    if (now.year() == pyNow.year and
            now.month() == pyNow.month and
            now.day() == pyNow.day and
            now.hour() == pyNow.hour and
            now.minute() == pyNow.minute and
            now.second() == pyNow.second):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    pyTm = pyNow.timetuple()
    print(colored('Testing dayOfWeek', 'cyan'))
    if now.dayOfWeek() == pyTm.tm_wday + 1:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing dayOfYear', 'cyan'))
    if now.dayOfYear() == pyTm.tm_yday:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing isDaylightSavings', 'cyan'))
    pyTm = time.localtime()
    if now.isDaylightSavings() == pyTm.tm_isdst:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing secondsPastMidnight', 'cyan'))
    if now.secondsPastMidnight() == pyTm.tm_hour * 3600 + pyTm.tm_min * 60 + pyTm.tm_sec:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

####################################################################################
if __name__ == '__main__':
    doTest()