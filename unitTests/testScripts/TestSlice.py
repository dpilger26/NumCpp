import numpy as np
from termcolor import colored
import sys
sys.path.append(r'../build/x64/Release')
import NumC

####################################################################################
def doTest():
    print(colored('Testing Slice Class', 'magenta'))

    print(colored('Testing Default Constructor', 'cyan'))
    cSlice = NumC.Slice()
    if cSlice.start == 0 and cSlice.stop == 1 and cSlice.step == 1:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Stop Only Constructor', 'cyan'))
    stop = np.random.randint(0, 100, [1,]).item()
    cSlice = NumC.Slice(stop)
    if cSlice.start == 0 and cSlice.stop == stop and cSlice.step == 1:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Start/Stop Constructor', 'cyan'))
    start = np.random.randint(0, 100, [1,]).item()
    stop = np.random.randint(100, 200, [1,]).item()
    cSlice = NumC.Slice(start, stop)
    if cSlice.start == start and cSlice.stop == stop and cSlice.step == 1:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Start/Stop/Step Constructor', 'cyan'))
    start = np.random.randint(0, 100, [1,]).item()
    stop = np.random.randint(100, 200, [1,]).item()
    step = np.random.randint(0, 50, [1, ]).item()
    cSlice = NumC.Slice(start, stop, step)
    if cSlice.start == start and cSlice.stop == stop and cSlice.step == step:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Copy Constructor', 'cyan'))
    cSlice2 = NumC.Slice(cSlice)
    if cSlice2.start == cSlice.start and cSlice2.stop == cSlice.stop and cSlice2.step == cSlice.step:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Setting members', 'cyan'))
    start = np.random.randint(0, 100, [1,]).item()
    stop = np.random.randint(100, 200, [1,]).item()
    step = np.random.randint(0, 50, [1, ]).item()
    cSlice = NumC.Slice()
    cSlice.start = start
    cSlice.stop = stop
    cSlice.step = step
    if cSlice.start == start and cSlice.stop == stop and cSlice.step == step:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

####################################################################################
if __name__ == '__main__':
    doTest()