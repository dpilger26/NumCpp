import time
import numpy as np
from termcolor import colored
import sys
sys.path.append(r'../build/x64/Release')
import NumCpp

####################################################################################
def doTest():
    print(colored('Testing Timer Class', 'magenta'))

    SLEEP_TIME = np.random.randint(0, 10, [1,]).item()
    print(colored(f'Sleeping for {round(SLEEP_TIME * 1e6)} microseconds with default Constructor', 'cyan'))
    timer = NumCpp.Timer()
    timer.tic()
    time.sleep(SLEEP_TIME)
    elapsedTime = timer.toc() # microseconds

    if round(elapsedTime / 1e6) == SLEEP_TIME:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    SLEEP_TIME = np.random.randint(0, 10, [1,]).item()
    print(colored(f'Sleeping for {round(SLEEP_TIME * 1e6)} microseconds with Named Constructor', 'cyan'))
    timer = NumCpp.Timer('Python Test Case')
    timer.tic()
    time.sleep(SLEEP_TIME)
    elapsedTime = timer.toc() # microseconds

    if round(elapsedTime / 1e6) == SLEEP_TIME:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

####################################################################################
if __name__ == '__main__':
    doTest()
