import numpy as np
import vectormath
import os
import sys

import NumCppPy as NumCpp  # noqa E402

DECIMALS_TO_ROUND = 9


####################################################################################
def test_Vec2_default_constructor():
    assert NumCpp.Vec2()


####################################################################################
def test_Vec2_component_constructor():
    components = np.random.rand(2)
    vec2 = NumCpp.Vec2(*components)
    assert vec2.x == components[0].item()
    assert vec2.y == components[1].item()


####################################################################################
def test_Vec2_array_constructor():
    components = np.random.rand(2)
    shape = NumCpp.Shape(1, 2)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(components)
    vec2 = NumCpp.Vec2(cArray)
    assert vec2.x == components[0].item()
    assert vec2.y == components[1].item()


####################################################################################
def test_Vec2_angle():
    components1 = np.random.rand(2)
    components2 = np.random.rand(2)
    vec2_1py = vectormath.Vector2(*components1)
    vec2_2py = vectormath.Vector2(*components2)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_2cpp = NumCpp.Vec2(*components2)
    assert round(vec2_1py.angle(vec2_2py), DECIMALS_TO_ROUND) == \
        round(vec2_1cpp.angle(vec2_2cpp), DECIMALS_TO_ROUND)


####################################################################################
def test_Vec2_up():
    vec2 = NumCpp.Vec2.up()
    assert vec2.x == 0
    assert vec2.y == 1


####################################################################################
def test_Vec2_down():
    vec2 = NumCpp.Vec2.down()
    assert vec2.x == 0
    assert vec2.y == -1


####################################################################################
def test_Vec2_right():
    vec2 = NumCpp.Vec2.right()
    assert vec2.x == 1
    assert vec2.y == 0


####################################################################################
def test_Vec2_left():
    vec2 = NumCpp.Vec2.left()
    assert vec2.x == -1
    assert vec2.y == -0


####################################################################################
def test_Vec2_clampMagnitude():
    components = np.random.rand(2) + 10
    clampMag = np.random.randint(1, 10, [1, ]).item()
    vec2 = NumCpp.Vec2(*components)
    clampedVec = vec2.clampMagnitude(float(clampMag))
    assert np.round(clampedVec.norm(), DECIMALS_TO_ROUND) == clampMag
    assert np.round(vec2.dot(clampedVec) / vec2.norm() /
                    clampedVec.norm(), DECIMALS_TO_ROUND) == 1


####################################################################################
def test_Vec2_distance():
    components1 = np.random.rand(2)
    components2 = np.random.rand(2)
    vec2_1py = vectormath.Vector2(*components1)
    vec2_2py = vectormath.Vector2(*components2)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_2cpp = NumCpp.Vec2(*components2)
    assert round((vec2_2py - vec2_1py).length, DECIMALS_TO_ROUND) == \
        round(vec2_1cpp.distance(vec2_2cpp), DECIMALS_TO_ROUND)


####################################################################################
def test_Vec2_dot():
    components1 = np.random.rand(2)
    components2 = np.random.rand(2)
    vec2_1py = vectormath.Vector2(*components1)
    vec2_2py = vectormath.Vector2(*components2)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_2cpp = NumCpp.Vec2(*components2)
    assert round(vec2_1py.dot(vec2_2py), DECIMALS_TO_ROUND) == \
        round(vec2_1cpp.dot(vec2_2cpp), DECIMALS_TO_ROUND)


####################################################################################
def test_Vec2_norm():
    components = np.random.rand(2)
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    assert round(vec2py.length, DECIMALS_TO_ROUND) == \
        round(vec2cpp.norm(), DECIMALS_TO_ROUND)


####################################################################################
def test_Vec2_normalize():
    components = np.random.rand(2)
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    assert np.array_equal(np.round(vec2py.normalize(), DECIMALS_TO_ROUND),
                          np.round(vec2cpp.normalize().toNdArray().flatten(), DECIMALS_TO_ROUND))


####################################################################################
def test_Vec2_equality_operator():
    components = np.random.rand(2)
    vec2_1cpp = NumCpp.Vec2(*components)
    vec2_2cpp = NumCpp.Vec2(*components)
    assert vec2_1cpp == vec2_2cpp
    assert not (vec2_1cpp != vec2_2cpp)


####################################################################################
def test_Vec2_addition_assignment_operator():
    components1 = np.random.rand(2)
    components2 = np.random.rand(2)
    vec2_1py = vectormath.Vector2(*components1)
    vec2_2py = vectormath.Vector2(*components2)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_2cpp = NumCpp.Vec2(*components2)
    vec2_1cpp += vec2_2cpp
    assert np.array_equal(np.round(vec2_1py + vec2_2py, DECIMALS_TO_ROUND),
                          np.round(vec2_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND))

    components1 = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2_1py = vectormath.Vector2(*components1)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_1cpp += scaler
    assert np.array_equal(np.round(vec2_1py + scaler, DECIMALS_TO_ROUND),
                          np.round(vec2_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND))


####################################################################################
def test_Vec2_subtraction_assignment_operator():
    components1 = np.random.rand(2)
    components2 = np.random.rand(2)
    vec2_1py = vectormath.Vector2(*components1)
    vec2_2py = vectormath.Vector2(*components2)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_2cpp = NumCpp.Vec2(*components2)
    vec2_1cpp -= vec2_2cpp
    assert np.array_equal(np.round(vec2_1py - vec2_2py, DECIMALS_TO_ROUND),
                          np.round(vec2_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND))

    components1 = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2_1py = vectormath.Vector2(*components1)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_1cpp -= scaler
    assert np.array_equal(np.round(vec2_1py - scaler, DECIMALS_TO_ROUND),
                          np.round(vec2_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND))


####################################################################################
def test_Vec2_multiply_assignment_operator():
    components1 = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2_1py = vectormath.Vector2(*components1)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_1cpp *= scaler
    assert np.array_equal(np.round(vec2_1py * scaler, DECIMALS_TO_ROUND),
                          np.round(vec2_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND))


####################################################################################
def test_Vec2_division_assignment_operator():
    components1 = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2_1py = vectormath.Vector2(*components1)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_1cpp /= scaler
    assert np.array_equal(np.round(vec2_1py / scaler, DECIMALS_TO_ROUND),
                          np.round(vec2_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND))


####################################################################################
def test_Vec2_addition_operator():
    components1 = np.random.rand(2)
    components2 = np.random.rand(2)
    vec2_1py = vectormath.Vector2(*components1)
    vec2_2py = vectormath.Vector2(*components2)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_2cpp = NumCpp.Vec2(*components2)
    assert np.array_equal(np.round(vec2_1py + vec2_2py, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec2_addVec2(vec2_1cpp, vec2_2cpp)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))

    components = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    assert np.array_equal(np.round(vec2py + scaler, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec2_addVec2Scaler(vec2cpp, scaler)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))

    components = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    assert np.array_equal(np.round(vec2py + scaler, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec2_addScalerVec2(vec2cpp, scaler)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))


####################################################################################
def test_Vec2_subtraction_operator():
    components1 = np.random.rand(2)
    components2 = np.random.rand(2)
    vec2_1py = vectormath.Vector2(*components1)
    vec2_2py = vectormath.Vector2(*components2)
    vec2_1cpp = NumCpp.Vec2(*components1)
    vec2_2cpp = NumCpp.Vec2(*components2)
    assert np.array_equal(np.round(vec2_1py - vec2_2py, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec2_minusVec2(vec2_1cpp, vec2_2cpp)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))

    components = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    assert np.array_equal(np.round(vec2py - scaler, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec2_minusVec2Scaler(vec2cpp, scaler)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))

    components = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    assert np.array_equal(np.round(-vec2py + scaler, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec2_minusScalerVec2(vec2cpp, scaler)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))


####################################################################################
def test_Vec2_multiplication_operator():
    components = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    assert np.array_equal(np.round(vec2py * scaler, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec2_multVec2Scaler(vec2cpp, scaler)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))

    components = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    assert np.array_equal(np.round(vec2py * scaler, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec2_multScalerVec2(vec2cpp, scaler)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))


####################################################################################
def test_Vec2_division_operator():
    components = np.random.rand(2)
    scaler = np.random.rand(1).item()
    vec2py = vectormath.Vector2(*components)
    vec2cpp = NumCpp.Vec2(*components)
    assert np.array_equal(np.round(vec2py / scaler, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec2_divVec2Scaler(vec2cpp, scaler)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))


####################################################################################
def test_Vec2_print():
    components = np.random.rand(2)
    vec2cpp = NumCpp.Vec2(*components)
    NumCpp.Vec2_print(vec2cpp)


####################################################################################
def test_Vec3_default_constructor():
    assert NumCpp.Vec3()


####################################################################################
def test_Vec3_component_constructor():
    components = np.random.rand(3)
    vec2 = NumCpp.Vec3(*components)
    assert vec2.x == components[0].item()
    assert vec2.y == components[1].item()
    assert vec2.z == components[2].item()


####################################################################################
def test_Vec3_array_constructor():
    components = np.random.rand(3)
    shape = NumCpp.Shape(1, 3)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(components)
    vec3 = NumCpp.Vec3(cArray)
    assert vec3.x == components[0].item()
    assert vec3.y == components[1].item()
    assert vec3.z == components[2].item()


####################################################################################
def test_Vec3_angle():
    components1 = np.random.rand(3)
    components2 = np.random.rand(3)
    vec3_1py = vectormath.Vector3(*components1)
    vec3_2py = vectormath.Vector3(*components2)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_2cpp = NumCpp.Vec3(*components2)
    assert (round(vec3_1py.angle(vec3_2py), DECIMALS_TO_ROUND) ==
            round(vec3_1cpp.angle(vec3_2cpp), DECIMALS_TO_ROUND))


####################################################################################
def test_Vec3_up():
    vec3 = NumCpp.Vec3.up()
    assert vec3.x == 0
    assert vec3.y == 1
    assert vec3.z == 0


####################################################################################
def test_Vec3_down():
    vec3 = NumCpp.Vec3.down()
    assert vec3.x == 0
    assert vec3.y == -1
    assert vec3.z == 0


####################################################################################
def test_Vec3_right():
    vec3 = NumCpp.Vec3.right()
    assert vec3.x == 1
    assert vec3.y == 0
    assert vec3.z == 0


####################################################################################
def test_Vec3_left():
    vec3 = NumCpp.Vec3.left()
    assert vec3.x == -1
    assert vec3.y == -0
    assert vec3.z == 0


####################################################################################
def test_Vec3_forward():
    vec3 = NumCpp.Vec3.forward()
    assert vec3.x == 0
    assert vec3.y == 0
    assert vec3.z == 1


####################################################################################
def test_Vec3_back():
    vec3 = NumCpp.Vec3.back()
    assert vec3.x == 0
    assert vec3.y == -0
    assert vec3.z == -1


####################################################################################
def test_Vec3_clampedVec():
    components = np.random.rand(3) + 10
    clampMag = np.random.randint(1, 10, [1, ]).item()
    vec3 = NumCpp.Vec3(*components)
    clampedVec = vec3.clampMagnitude(float(clampMag))
    assert np.round(clampedVec.norm(), DECIMALS_TO_ROUND) == clampMag
    assert np.round(vec3.dot(clampedVec) / vec3.norm() /
                    clampedVec.norm(), DECIMALS_TO_ROUND) == 1


####################################################################################
def test_Vec3_distance():
    components1 = np.random.rand(3)
    components2 = np.random.rand(3)
    vec3_1py = vectormath.Vector3(*components1)
    vec3_2py = vectormath.Vector3(*components2)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_2cpp = NumCpp.Vec3(*components2)
    assert (round((vec3_2py - vec3_1py).length, DECIMALS_TO_ROUND) ==
            round(vec3_1cpp.distance(vec3_2cpp), DECIMALS_TO_ROUND))


####################################################################################
def test_Vec3_dot():
    components1 = np.random.rand(3)
    components2 = np.random.rand(3)
    vec3_1py = vectormath.Vector3(*components1)
    vec3_2py = vectormath.Vector3(*components2)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_2cpp = NumCpp.Vec3(*components2)
    assert (round(vec3_1py.dot(vec3_2py), DECIMALS_TO_ROUND) ==
            round(vec3_1cpp.dot(vec3_2cpp), DECIMALS_TO_ROUND))


####################################################################################
def test_Vec3_cross():
    components1 = np.random.rand(3)
    components2 = np.random.rand(3)
    vec3_1py = vectormath.Vector3(*components1)
    vec3_2py = vectormath.Vector3(*components2)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_2cpp = NumCpp.Vec3(*components2)
    assert np.array_equal(np.round(vec3_1py.cross(vec3_2py), DECIMALS_TO_ROUND),
                          np.round(vec3_1cpp.cross(vec3_2cpp).toNdArray().flatten(), DECIMALS_TO_ROUND))


####################################################################################
def test_Vec3_norm():
    components = np.random.rand(3)
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    assert (round(vec3py.length, DECIMALS_TO_ROUND) ==
            round(vec3cpp.norm(), DECIMALS_TO_ROUND))


####################################################################################
def test_Vec3_normalize():
    components = np.random.rand(3)
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    assert np.array_equal(np.round(vec3py.normalize(), DECIMALS_TO_ROUND),
                          np.round(vec3cpp.normalize().toNdArray().flatten(), DECIMALS_TO_ROUND))


####################################################################################
def test_Vec3_equality_operator():
    components = np.random.rand(3)
    vec3_1cpp = NumCpp.Vec3(*components)
    vec3_2cpp = NumCpp.Vec3(*components)
    assert vec3_1cpp == vec3_2cpp
    assert not (vec3_1cpp != vec3_2cpp)


####################################################################################
def test_Vec3_addition_assignment_operator():
    components1 = np.random.rand(3)
    components2 = np.random.rand(3)
    vec3_1py = vectormath.Vector3(*components1)
    vec3_2py = vectormath.Vector3(*components2)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_2cpp = NumCpp.Vec3(*components2)
    vec3_1cpp += vec3_2cpp
    assert np.array_equal(np.round(vec3_1py + vec3_2py, DECIMALS_TO_ROUND),
                          np.round(vec3_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND))

    components1 = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3_1py = vectormath.Vector3(*components1)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_1cpp += scaler
    assert np.array_equal(np.round(vec3_1py + scaler, DECIMALS_TO_ROUND),
                          np.round(vec3_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND))


####################################################################################
def test_Vec3_subtraction_assignment_operator():
    components1 = np.random.rand(3)
    components2 = np.random.rand(3)
    vec3_1py = vectormath.Vector3(*components1)
    vec3_2py = vectormath.Vector3(*components2)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_2cpp = NumCpp.Vec3(*components2)
    vec3_1cpp -= vec3_2cpp
    assert np.array_equal(np.round(vec3_1py - vec3_2py, DECIMALS_TO_ROUND),
                          np.round(vec3_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND))

    components1 = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3_1py = vectormath.Vector3(*components1)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_1cpp -= scaler
    assert np.array_equal(np.round(vec3_1py - scaler, DECIMALS_TO_ROUND),
                          np.round(vec3_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND))


####################################################################################
def test_Vec3_multiplication_assignment_operator():
    components1 = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3_1py = vectormath.Vector3(*components1)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_1cpp *= scaler
    assert np.array_equal(np.round(vec3_1py * scaler, DECIMALS_TO_ROUND),
                          np.round(vec3_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND))


####################################################################################
def test_Vec3_division_assignment_operator():
    components1 = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3_1py = vectormath.Vector3(*components1)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_1cpp /= scaler
    assert np.array_equal(np.round(vec3_1py / scaler, DECIMALS_TO_ROUND),
                          np.round(vec3_1cpp.toNdArray().flatten(), DECIMALS_TO_ROUND))


####################################################################################
def test_Vec3_addition_operator():
    components1 = np.random.rand(3)
    components2 = np.random.rand(3)
    vec3_1py = vectormath.Vector3(*components1)
    vec3_2py = vectormath.Vector3(*components2)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_2cpp = NumCpp.Vec3(*components2)
    assert np.array_equal(np.round(vec3_1py + vec3_2py, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec3_addVec3(vec3_1cpp, vec3_2cpp)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))

    components = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    assert np.array_equal(np.round(vec3py + scaler, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec3_addVec3Scaler(vec3cpp, scaler)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))

    components = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    assert np.array_equal(np.round(vec3py + scaler, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec3_addScalerVec3(vec3cpp, scaler)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))


####################################################################################
def test_Vec3_subtraction_operator():
    components1 = np.random.rand(3)
    components2 = np.random.rand(3)
    vec3_1py = vectormath.Vector3(*components1)
    vec3_2py = vectormath.Vector3(*components2)
    vec3_1cpp = NumCpp.Vec3(*components1)
    vec3_2cpp = NumCpp.Vec3(*components2)
    assert np.array_equal(np.round(vec3_1py - vec3_2py, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec3_minusVec3(vec3_1cpp, vec3_2cpp)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))

    components = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    assert np.array_equal(np.round(vec3py - scaler, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec3_minusVec3Scaler(vec3cpp, scaler)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))

    components = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    assert np.array_equal(np.round(-vec3py + scaler, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec3_minusScalerVec3(vec3cpp, scaler)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))


####################################################################################
def test_Vec3_multiplication_operator():
    components = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    assert np.array_equal(np.round(vec3py * scaler, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec3_multVec3Scaler(vec3cpp, scaler)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))

    components = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    assert np.array_equal(np.round(vec3py * scaler, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec3_multScalerVec3(vec3cpp, scaler)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))


####################################################################################
def test_Vec3_division_operator():
    components = np.random.rand(3)
    scaler = np.random.rand(1).item()
    vec3py = vectormath.Vector3(*components)
    vec3cpp = NumCpp.Vec3(*components)
    assert np.array_equal(np.round(vec3py / scaler, DECIMALS_TO_ROUND),
                          np.round((NumCpp.Vec3_divVec3Scaler(vec3cpp, scaler)).toNdArray().flatten(),
                                   DECIMALS_TO_ROUND))


####################################################################################
def test_Vec3_print():
    components = np.random.rand(3)
    vec3cpp = NumCpp.Vec3(*components)
    NumCpp.Vec3_print(vec3cpp)
