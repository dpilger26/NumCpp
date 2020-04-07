import numpy as np
from termcolor import colored
import os
import sys
if sys.platform == 'linux':
    sys.path.append(r'../lib')
else:
    sys.path.append(os.path.abspath('../build/x64/Release'))
import NumCpp


####################################################################################
def doTest():
    print(colored('Testing Timer Class', 'magenta'))

    SLEEP_TIME = int(np.random.randint(0, 10, [1, ]).item() * 1e6)  # microseconds
    print(colored(f'Sleeping for {SLEEP_TIME} microseconds with default Constructor', 'cyan'))
    timer = NumCpp.Timer()
    timer.tic()
    timer.sleep(SLEEP_TIME)
    elapsedTime = timer.toc(True)  # microseconds

    if np.abs(elapsedTime - SLEEP_TIME) < 0.1e6:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    SLEEP_TIME = int(np.random.randint(0, 10, [1, ]).item() * 1e6)  # microseconds
    print(colored(f'Sleeping for {SLEEP_TIME} microseconds with Named Constructor', 'cyan'))
    timer = NumCpp.Timer('Python Test Case')
    timer.tic()
    timer.sleep(SLEEP_TIME)
    elapsedTime = timer.toc(True)  # microseconds

    if np.abs(elapsedTime - SLEEP_TIME) < 0.1e6:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))


####################################################################################
if __name__ == '__main__':
    doTest()
