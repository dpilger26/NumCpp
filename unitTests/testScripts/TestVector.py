import numpy as np
from termcolor import colored
import vectormath
import sys
if sys.platform == 'linux':
    sys.path.append(r'../lib')
else:
    sys.path.append(r'../build/x64/Release')
import NumCpp

DECIMALS_TO_ROUND = 9


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
    if (round(vec2_1py.angle(vec2_2py), DECIMALS_TO_ROUND) ==
            round(vec2_1cpp.angle(vec2_2cpp), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing up', 'cyan'))
    vec2 = NumCpp.Vec2.up()
    if vec2.x == 0 and vec2.y == 1:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing down', 'cyan'))
    vec2 = NumCpp.Vec2.down()
    if vec2.x == 0 and vec2.y == -1:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing right', 'cyan'))
    vec2 = NumCpp.Vec2.right()
    if vec2.x == 1 and vec2.y == 0:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing down', 'cyan'))
    vec2 = NumCpp.Vec2.left()
    if vec2.x == -1 and vec2.y == -0:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing clampMagnitude', 'cyan'))
    components = np.random.rand(2) + 10
    clampMag = np.random.randint(1, 10, [1, ]).item()
    vec2 = NumCpp.Vec2(*components)
    clampedVec = vec2.clampMagnitude(float(clampMag))
    if (np.round(clampedVec.norm(), DECIMALS_TO_ROUND) == clampMag and
            np.round(vec2.dot(clampedVec) / vec2.norm() / clampedVec.norm(), DECIMALS_TO_ROUND) == 1):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing distance', 'cyan'))
    components1 = np.random.rand(2)
    components2 = np.random.rand(2)
    vec2_1py = vectormath.Vector2(*components1)
    vec2_2py = vectormath.Vector2(*components2)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_2cpp = NumCpp.Vec2(*components2)
    if (round((vec2_2py - vec2_1py).length, DECIMALS_TO_ROUND) ==
            round(vec2_1cpp.distance(vec2_2cpp), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing dot', 'cyan'))
    components1 = np.random.rand(2)
    components2 = np.random.rand(2)
    vec2_1py = vectormath.Vector2(*components1)
    vec2_2py = vectormath.Vector2(*components2)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_2cpp = NumCpp.Vec2(*components2)
    if (round(vec2_1py.dot(vec2_2py), DECIMALS_TO_ROUND) ==
            round(vec2_1cpp.dot(vec2_2cpp), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing norm', 'cyan'))
    components = np.random.rand(2)
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    if (round(vec2py.length, DECIMALS_TO_ROUND) ==
            round(vec2cpp.norm(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing normalize', 'cyan'))
    components = np.random.rand(2)
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    if np.array_equal(np.round(vec2py.normalize(), DECIMALS_TO_ROUND),
                      np.round(vec2cpp.normalize().toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator==', 'cyan'))
    components = np.random.rand(2)
    vec2_1cpp = NumCpp.Vec2(*components)
    vec2_2cpp = NumCpp.Vec2(*components)
    if vec2_1cpp == vec2_2cpp:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator!=', 'cyan'))
    if not (vec2_1cpp != vec2_2cpp):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator+= Vec2', 'cyan'))
    components1 = np.random.rand(2)
    components2 = np.random.rand(2)
    vec2_1py = vectormath.Vector2(*components1)
    vec2_2py = vectormath.Vector2(*components2)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_2cpp = NumCpp.Vec2(*components2)
    vec2_1cpp += vec2_2cpp
    if np.array_equal(np.round(vec2_1py + vec2_2py, DECIMALS_TO_ROUND),
                      np.round(vec2_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator+= scaler', 'cyan'))
    components1 = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2_1py = vectormath.Vector2(*components1)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_1cpp += scaler
    if np.array_equal(np.round(vec2_1py + scaler, DECIMALS_TO_ROUND),
                      np.round(vec2_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator-=', 'cyan'))
    components1 = np.random.rand(2)
    components2 = np.random.rand(2)
    vec2_1py = vectormath.Vector2(*components1)
    vec2_2py = vectormath.Vector2(*components2)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_2cpp = NumCpp.Vec2(*components2)
    vec2_1cpp -= vec2_2cpp
    if np.array_equal(np.round(vec2_1py - vec2_2py, DECIMALS_TO_ROUND),
                      np.round(vec2_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator-= scaler', 'cyan'))
    components1 = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2_1py = vectormath.Vector2(*components1)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_1cpp -= scaler
    if np.array_equal(np.round(vec2_1py - scaler, DECIMALS_TO_ROUND),
                      np.round(vec2_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator*=', 'cyan'))
    components1 = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2_1py = vectormath.Vector2(*components1)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_1cpp *= scaler
    if np.array_equal(np.round(vec2_1py * scaler, DECIMALS_TO_ROUND),
                      np.round(vec2_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator/=', 'cyan'))
    components1 = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2_1py = vectormath.Vector2(*components1)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_1cpp /= scaler
    if np.array_equal(np.round(vec2_1py / scaler, DECIMALS_TO_ROUND),
                      np.round(vec2_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator+ Vec2', 'cyan'))
    components1 = np.random.rand(2)
    components2 = np.random.rand(2)
    vec2_1py = vectormath.Vector2(*components1)
    vec2_2py = vectormath.Vector2(*components2)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_2cpp = NumCpp.Vec2(*components2)
    if np.array_equal(np.round(vec2_1py + vec2_2py, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec2_addVec2(vec2_1cpp, vec2_2cpp)).toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator+ scaler', 'cyan'))
    components = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    if np.array_equal(np.round(vec2py + scaler, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec2_addVec2Scaler(vec2cpp, scaler)).toNdArray().flatten(),
                               DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing scaler operator+ Vec2', 'cyan'))
    components = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    if np.array_equal(np.round(vec2py + scaler, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec2_addScalerVec2(vec2cpp, scaler)).toNdArray().flatten(),
                               DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator- Vec2', 'cyan'))
    components1 = np.random.rand(2)
    components2 = np.random.rand(2)
    vec2_1py = vectormath.Vector2(*components1)
    vec2_2py = vectormath.Vector2(*components2)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_2cpp = NumCpp.Vec2(*components2)
    if np.array_equal(np.round(vec2_1py - vec2_2py, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec2_minusVec2(vec2_1cpp, vec2_2cpp)).toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator- scaler', 'cyan'))
    components = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    if np.array_equal(np.round(vec2py - scaler, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec2_minusVec2Scaler(vec2cpp, scaler)).toNdArray().flatten(),
                               DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing scaler operator- Vec2', 'cyan'))
    components = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    if np.array_equal(np.round(-vec2py + scaler, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec2_minusScalerVec2(vec2cpp, scaler)).toNdArray().flatten(),
                               DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator* scaler', 'cyan'))
    components = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    if np.array_equal(np.round(vec2py * scaler, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec2_multVec2Scaler(vec2cpp, scaler)).toNdArray().flatten(),
                               DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing scaler operator* Vec2', 'cyan'))
    components = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    if np.array_equal(np.round(vec2py * scaler, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec2_multScalerVec2(vec2cpp, scaler)).toNdArray().flatten(),
                               DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator/ scaler', 'cyan'))
    components = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    if np.array_equal(np.round(vec2py / scaler, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec2_divVec2Scaler(vec2cpp, scaler)).toNdArray().flatten(),
                               DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing print', 'cyan'))
    NumCpp.Vec2_print(vec2cpp)


####################################################################################
def testVec3():
    print(colored('Testing Vec3 Class', 'magenta'))

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
    if (round(vec3_1py.angle(vec3_2py), DECIMALS_TO_ROUND) ==
            round(vec3_1cpp.angle(vec3_2cpp), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing up', 'cyan'))
    vec3 = NumCpp.Vec3.up()
    if vec3.x == 0 and vec3.y == 1 and vec3.z == 0:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing down', 'cyan'))
    vec3 = NumCpp.Vec3.down()
    if vec3.x == 0 and vec3.y == -1 and vec3.z == 0:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing right', 'cyan'))
    vec3 = NumCpp.Vec3.right()
    if vec3.x == 1 and vec3.y == 0 and vec3.z == 0:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing down', 'cyan'))
    vec3 = NumCpp.Vec3.left()
    if vec3.x == -1 and vec3.y == -0 and vec3.z == 0:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing forward', 'cyan'))
    vec3 = NumCpp.Vec3.forward()
    if vec3.x == 0 and vec3.y == 0 and vec3.z == 1:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing back', 'cyan'))
    vec3 = NumCpp.Vec3.back()
    if vec3.x == 0 and vec3.y == -0 and vec3.z == -1:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing clampMagnitude', 'cyan'))
    components = np.random.rand(3) + 10
    clampMag = np.random.randint(1, 10, [1, ]).item()
    vec3 = NumCpp.Vec3(*components)
    clampedVec = vec3.clampMagnitude(float(clampMag))
    if (np.round(clampedVec.norm(), DECIMALS_TO_ROUND) == clampMag and
            np.round(vec3.dot(clampedVec) / vec3.norm() / clampedVec.norm(), DECIMALS_TO_ROUND) == 1):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing distance', 'cyan'))
    components1 = np.random.rand(3)
    components2 = np.random.rand(3)
    vec3_1py = vectormath.Vector3(*components1)
    vec3_2py = vectormath.Vector3(*components2)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_2cpp = NumCpp.Vec3(*components2)
    if (round((vec3_2py - vec3_1py).length, DECIMALS_TO_ROUND) ==
            round(vec3_1cpp.distance(vec3_2cpp), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing dot', 'cyan'))
    components1 = np.random.rand(3)
    components2 = np.random.rand(3)
    vec3_1py = vectormath.Vector3(*components1)
    vec3_2py = vectormath.Vector3(*components2)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_2cpp = NumCpp.Vec3(*components2)
    if (round(vec3_1py.dot(vec3_2py), DECIMALS_TO_ROUND) ==
            round(vec3_1cpp.dot(vec3_2cpp), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cross', 'cyan'))
    components1 = np.random.rand(3)
    components2 = np.random.rand(3)
    vec3_1py = vectormath.Vector3(*components1)
    vec3_2py = vectormath.Vector3(*components2)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_2cpp = NumCpp.Vec3(*components2)
    if np.array_equal(np.round(vec3_1py.cross(vec3_2py), DECIMALS_TO_ROUND),
                      np.round(vec3_1cpp.cross(vec3_2cpp).toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing norm', 'cyan'))
    components = np.random.rand(3)
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    if (round(vec3py.length, DECIMALS_TO_ROUND) ==
            round(vec3cpp.norm(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing normalize', 'cyan'))
    components = np.random.rand(3)
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    if np.array_equal(np.round(vec3py.normalize(), DECIMALS_TO_ROUND),
                      np.round(vec3cpp.normalize().toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator==', 'cyan'))
    components = np.random.rand(3)
    vec3_1cpp = NumCpp.Vec3(*components)
    vec3_2cpp = NumCpp.Vec3(*components)
    if vec3_1cpp == vec3_2cpp:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator!=', 'cyan'))
    if not (vec3_1cpp != vec3_2cpp):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator+= Vec3', 'cyan'))
    components1 = np.random.rand(3)
    components2 = np.random.rand(3)
    vec3_1py = vectormath.Vector3(*components1)
    vec3_2py = vectormath.Vector3(*components2)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_2cpp = NumCpp.Vec3(*components2)
    vec3_1cpp += vec3_2cpp
    if np.array_equal(np.round(vec3_1py + vec3_2py, DECIMALS_TO_ROUND),
                      np.round(vec3_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator+= scaler', 'cyan'))
    components1 = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3_1py = vectormath.Vector3(*components1)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_1cpp += scaler
    if np.array_equal(np.round(vec3_1py + scaler, DECIMALS_TO_ROUND),
                      np.round(vec3_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator-=', 'cyan'))
    components1 = np.random.rand(3)
    components2 = np.random.rand(3)
    vec3_1py = vectormath.Vector3(*components1)
    vec3_2py = vectormath.Vector3(*components2)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_2cpp = NumCpp.Vec3(*components2)
    vec3_1cpp -= vec3_2cpp
    if np.array_equal(np.round(vec3_1py - vec3_2py, DECIMALS_TO_ROUND),
                      np.round(vec3_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator-= scaler', 'cyan'))
    components1 = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3_1py = vectormath.Vector3(*components1)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_1cpp -= scaler
    if np.array_equal(np.round(vec3_1py - scaler, DECIMALS_TO_ROUND),
                      np.round(vec3_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator*=', 'cyan'))
    components1 = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3_1py = vectormath.Vector3(*components1)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_1cpp *= scaler
    if np.array_equal(np.round(vec3_1py * scaler, DECIMALS_TO_ROUND),
                      np.round(vec3_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator/=', 'cyan'))
    components1 = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3_1py = vectormath.Vector3(*components1)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_1cpp /= scaler
    if np.array_equal(np.round(vec3_1py / scaler, DECIMALS_TO_ROUND),
                      np.round(vec3_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator+ Vec3', 'cyan'))
    components1 = np.random.rand(3)
    components2 = np.random.rand(3)
    vec3_1py = vectormath.Vector3(*components1)
    vec3_2py = vectormath.Vector3(*components2)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_2cpp = NumCpp.Vec3(*components2)
    if np.array_equal(np.round(vec3_1py + vec3_2py, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec3_addVec3(vec3_1cpp, vec3_2cpp)).toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator+ scaler', 'cyan'))
    components = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    if np.array_equal(np.round(vec3py + scaler, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec3_addVec3Scaler(vec3cpp, scaler)).toNdArray().flatten(),
                               DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing scaler operator+ Vec3', 'cyan'))
    components = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    if np.array_equal(np.round(vec3py + scaler, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec3_addScalerVec3(vec3cpp, scaler)).toNdArray().flatten(),
                               DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator- Vec3', 'cyan'))
    components1 = np.random.rand(3)
    components2 = np.random.rand(3)
    vec3_1py = vectormath.Vector3(*components1)
    vec3_2py = vectormath.Vector3(*components2)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_2cpp = NumCpp.Vec3(*components2)
    if np.array_equal(np.round(vec3_1py - vec3_2py, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec3_minusVec3(vec3_1cpp, vec3_2cpp)).toNdArray().flatten(), DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator- scaler', 'cyan'))
    components = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    if np.array_equal(np.round(vec3py - scaler, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec3_minusVec3Scaler(vec3cpp, scaler)).toNdArray().flatten(),
                               DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing scaler operator- Vec3', 'cyan'))
    components = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    if np.array_equal(np.round(-vec3py + scaler, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec3_minusScalerVec3(vec3cpp, scaler)).toNdArray().flatten(),
                               DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator* scaler', 'cyan'))
    components = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    if np.array_equal(np.round(vec3py * scaler, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec3_multVec3Scaler(vec3cpp, scaler)).toNdArray().flatten(),
                               DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing scaler operator* Vec3', 'cyan'))
    components = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    if np.array_equal(np.round(vec3py * scaler, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec3_multScalerVec3(vec3cpp, scaler)).toNdArray().flatten(),
                               DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator/ scaler', 'cyan'))
    components = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    if np.array_equal(np.round(vec3py / scaler, DECIMALS_TO_ROUND),
                      np.round((NumCpp.Vec3_divVec3Scaler(vec3cpp, scaler)).toNdArray().flatten(),
                               DECIMALS_TO_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing print', 'cyan'))
    NumCpp.Vec3_print(vec3cpp)


####################################################################################
if __name__ == '__main__':
    doTest()
