import numpy as np
from termcolor import colored
import vectormath
import sys
if sys.platform == 'linux':
    sys.path.append(r'../lib')
else:
    sys.path.append(r'../build/x64/Release')
import NumCpp


####################################################################################
def doTest():
    print(colored('Testing Vector Module', 'magenta'))

    testVec2()
    testVec3()


####################################################################################
def testVec2():
    print(colored('Testing Vec2 Class', 'magenta'))

    print(colored('Testing Default Constructor', 'cyan'))
    NumCpp.Vec2()
    print(colored('\tPASS', 'green'))

    print(colored('Testing Value Constructor', 'cyan'))
    components = np.random.rand(2)
    vec2 = NumCpp.Vec2(*components)
    if (vec2.x == components[0].item() and
            vec2.y == components[1].item()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing NdArray Constructor', 'cyan'))
    components = np.random.rand(2)
    shape = NumCpp.Shape(1, 2)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(components)
    vec2 = NumCpp.Vec2(cArray)
    if (vec2.x == components[0].item() and
            vec2.y == components[1].item()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing angle', 'cyan'))
    components1 = np.random.rand(2)
    components2 = np.random.rand(2)
    vec2_1py = vectormath.Vector2(*components1)
    vec2_2py = vectormath.Vector2(*components2)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_2cpp = NumCpp.Vec2(*components2)

    if vec2_1py.angle(vec2_2py) == vec2_1cpp.angle(vec2_2cpp):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))


####################################################################################
def testVec3():
    print(colored('Testing Vec2 Class', 'magenta'))

    print(colored('Testing Default Constructor', 'cyan'))
    NumCpp.Vec3()
    print(colored('\tPASS', 'green'))

    print(colored('Testing Value Constructor', 'cyan'))
    components = np.random.rand(3)
    vec2 = NumCpp.Vec3(*components)
    if (vec2.x == components[0].item() and
            vec2.y == components[1].item() and
            vec2.z == components[2].item()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing NdArray Constructor', 'cyan'))
    components = np.random.rand(3)
    shape = NumCpp.Shape(1, 3)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(components)
    vec3 = NumCpp.Vec3(cArray)
    if (vec3.x == components[0].item() and
            vec3.y == components[1].item() and
            vec3.z == components[2].item()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing angle', 'cyan'))
    components1 = np.random.rand(3)
    components2 = np.random.rand(3)
    vec3_1py = vectormath.Vector3(*components1)
    vec3_2py = vectormath.Vector3(*components2)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_2cpp = NumCpp.Vec3(*components2)

    if vec3_1py.angle(vec3_2py) == vec3_1cpp.angle(vec3_2cpp):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))


####################################################################################
if __name__ == '__main__':
    doTest()
