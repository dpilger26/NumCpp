import numpy as np
from termcolor import colored
import sys
sys.path.append(r'../build/x64/Release')
import NumC

####################################################################################
def doTest():
    print(colored('Testing Utils', 'magenta'))
    print(colored('Testing num2str', 'cyan'))
    value = np.random.randint(1, 100, [1, ], dtype=np.int8).item()
    if NumC.num2str(value) == str(value):
        print(colored('\tPASS int8', 'green'))
    else:
        print(colored('\tFAIL int8', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.int16).item()
    if NumC.num2str(value) == str(value):
        print(colored('\tPASS int16', 'green'))
    else:
        print(colored('\tFAIL int16', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.int32).item()
    if NumC.num2str(value) == str(value):
        print(colored('\tPASS int32', 'green'))
    else:
        print(colored('\tFAIL int32', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.int64).item()
    if NumC.num2str(value) == str(value):
        print(colored('\tPASS int64', 'green'))
    else:
        print(colored('\tFAIL int64', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.uint8).item()
    if NumC.num2str(value) == str(value):
        print(colored('\tPASS uint8', 'green'))
    else:
        print(colored('\tFAIL uint8', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.uint16).item()
    if NumC.num2str(value) == str(value):
        print(colored('\tPASS uint16', 'green'))
    else:
        print(colored('\tFAIL uint16', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.uint32).item()
    if NumC.num2str(value) == str(value):
        print(colored('\tPASS uint32', 'green'))
    else:
        print(colored('\tFAIL uint32', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.uint64).item()
    if NumC.num2str(value) == str(value):
        print(colored('\tPASS uint64', 'green'))
    else:
        print(colored('\tFAIL uint64', 'red'))

    print(colored('Testing sqr', 'cyan'))
    value = np.random.randint(1, 100, [1, ], dtype=np.int8).item()
    if NumC.sqr(value) == value ** 2:
        print(colored('\tPASS int8', 'green'))
    else:
        print(colored('\tFAIL int8', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.int16).item()
    if NumC.sqr(value) == value ** 2:
        print(colored('\tPASS int16', 'green'))
    else:
        print(colored('\tFAIL int16', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.int32).item()
    if NumC.sqr(value) == value ** 2:
        print(colored('\tPASS int32', 'green'))
    else:
        print(colored('\tFAIL int32', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.int64).item()
    if NumC.sqr(value) == value ** 2:
        print(colored('\tPASS int64', 'green'))
    else:
        print(colored('\tFAIL int64', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.uint8).item()
    if NumC.sqr(value) == value ** 2:
        print(colored('\tPASS uint8', 'green'))
    else:
        print(colored('\tFAIL uint8', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.uint16).item()
    if NumC.sqr(value) == value ** 2:
        print(colored('\tPASS uint16', 'green'))
    else:
        print(colored('\tFAIL uint16', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.uint32).item()
    if NumC.sqr(value) == value ** 2:
        print(colored('\tPASS uint32', 'green'))
    else:
        print(colored('\tFAIL uint32', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.uint64).item()
    if NumC.sqr(value) == value ** 2:
        print(colored('\tPASS uint64', 'green'))
    else:
        print(colored('\tFAIL uint64', 'red'))

    value = np.random.randint(1, 100, [1, ]).astype(np.double).item()
    if NumC.sqr(value) == value ** 2:
        print(colored('\tPASS double', 'green'))
    else:
        print(colored('\tFAIL double', 'red'))

    value = np.random.randint(1, 100, [1, ]).astype(np.float32).item()
    if NumC.sqr(value) == value ** 2:
        print(colored('\tPASS float', 'green'))
    else:
        print(colored('\tFAIL float', 'red'))

    print(colored('Testing cube', 'cyan'))
    value = np.random.randint(1, 100, [1, ], dtype=np.int8).item()
    if NumC.cube(value) == value ** 3:
        print(colored('\tPASS int8', 'green'))
    else:
        print(colored('\tFAIL int8', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.int16).item()
    if NumC.cube(value) == value ** 3:
        print(colored('\tPASS int16', 'green'))
    else:
        print(colored('\tFAIL int16', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.int32).item()
    if NumC.cube(value) == value ** 3:
        print(colored('\tPASS int32', 'green'))
    else:
        print(colored('\tFAIL int32', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.int64).item()
    if NumC.cube(value) == value ** 3:
        print(colored('\tPASS int64', 'green'))
    else:
        print(colored('\tFAIL int64', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.uint8).item()
    if NumC.cube(value) == value ** 3:
        print(colored('\tPASS uint8', 'green'))
    else:
        print(colored('\tFAIL uint8', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.uint16).item()
    if NumC.cube(value) == value ** 3:
        print(colored('\tPASS uint16', 'green'))
    else:
        print(colored('\tFAIL uint16', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.uint32).item()
    if NumC.cube(value) == value ** 3:
        print(colored('\tPASS uint32', 'green'))
    else:
        print(colored('\tFAIL uint32', 'red'))

    value = np.random.randint(1, 100, [1, ], dtype=np.uint64).item()
    if NumC.cube(value) == value ** 3:
        print(colored('\tPASS uint64', 'green'))
    else:
        print(colored('\tFAIL uint64', 'red'))

    value = np.random.randint(1, 100, [1, ]).astype(np.double).item()
    if NumC.cube(value) == value ** 3:
        print(colored('\tPASS double', 'green'))
    else:
        print(colored('\tFAIL double', 'red'))

    value = np.random.randint(1, 100, [1, ]).astype(np.float32).item()
    if NumC.cube(value) == value ** 3:
        print(colored('\tPASS float', 'green'))
    else:
        print(colored('\tFAIL float', 'red'))

    print(colored('Testing power', 'cyan'))
    value = np.random.randint(1, 10, [1, ], dtype=np.int8).item()
    power = np.random.randint(1, 10, dtype=np.uint8).item()
    if NumC.power(value, power) == value ** power:
        print(colored('\tPASS int8', 'green'))
    else:
        print(colored('\tFAIL int8', 'red'))

    value = np.random.randint(1, 10, [1, ], dtype=np.int16).item()
    if NumC.power(value, power) == value ** power:
        print(colored('\tPASS int16', 'green'))
    else:
        print(colored('\tFAIL int16', 'red'))

    value = np.random.randint(1, 10, [1, ], dtype=np.int32).item()
    if NumC.power(value, power) == value ** power:
        print(colored('\tPASS int32', 'green'))
    else:
        print(colored('\tFAIL int32', 'red'))

    value = np.random.randint(1, 10, [1, ], dtype=np.int64).item()
    if NumC.power(value, power) == value ** power:
        print(colored('\tPASS int64', 'green'))
    else:
        print(colored('\tFAIL int64', 'red'))

    value = np.random.randint(1, 10, [1, ], dtype=np.uint8).item()
    if NumC.power(value, power) == value ** power:
        print(colored('\tPASS uint8', 'green'))
    else:
        print(colored('\tFAIL uint8', 'red'))

    value = np.random.randint(1, 10, [1, ], dtype=np.uint16).item()
    if NumC.power(value, power) == value ** power:
        print(colored('\tPASS uint16', 'green'))
    else:
        print(colored('\tFAIL uint16', 'red'))

    value = np.random.randint(1, 10, [1, ], dtype=np.uint32).item()
    if NumC.power(value, power) == value ** power:
        print(colored('\tPASS uint32', 'green'))
    else:
        print(colored('\tFAIL uint32', 'red'))

    value = np.random.randint(1, 10, [1, ], dtype=np.uint64).item()
    if NumC.power(value, power) == value ** power:
        print(colored('\tPASS uint64', 'green'))
    else:
        print(colored('\tFAIL uint64', 'red'))

    value = np.random.randint(1, 10, [1, ]).astype(np.double).item()
    if NumC.power(value, power) == value ** power:
        print(colored('\tPASS double', 'green'))
    else:
        print(colored('\tFAIL double', 'red'))

    value = np.random.randint(1, 10, [1, ]).astype(np.float32).item()
    if NumC.power(value, power) == value ** power:
        print(colored('\tPASS float', 'green'))
    else:
        print(colored('\tFAIL float', 'red'))

####################################################################################
if __name__ == '__main__':
    doTest()