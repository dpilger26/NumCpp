import numpy as np
import os
import sys
sys.path.append(os.path.abspath(r'../lib'))
import NumCppPy as NumCpp  # noqa E402


####################################################################################
def test_seed():
    np.random.seed(666)


####################################################################################
def test_quaternion():
    assert NumCpp.Quaternion()

    quat = np.random.randint(1, 10, [4, ])
    unitQuat = quat / np.linalg.norm(quat)
    quat = NumCpp.Quaternion(quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item())
    assert (quat.i() == unitQuat[0].item() and
            quat.j() == unitQuat[1].item() and
            quat.k() == unitQuat[2].item() and
            quat.s() == unitQuat[3].item())

    quat = np.random.randint(1, 10, [4, ])
    unitQuat = quat / np.linalg.norm(quat)
    cArray = NumCpp.NdArray(1, 4)
    cArray.setArray(quat)
    quat = NumCpp.Quaternion(cArray)
    assert (quat.i() == unitQuat[0].item() and
            quat.j() == unitQuat[1].item() and
            quat.k() == unitQuat[2].item() and
            quat.s() == unitQuat[3].item())

    roll = np.random.rand(1).item() * np.pi * 2 - np.pi  # [-pi, pi]
    pitch = np.random.rand(1).item() * np.pi - np.pi / 2  # [-pi/2, pi/2]
    yaw = np.random.rand(1).item() * np.pi * 2 - np.pi  # [-pi, pi]
    quat = NumCpp.Quaternion(roll, pitch, yaw)
    assert (np.round(quat.roll(), 10) == np.round(roll, 10) and
            np.round(quat.pitch(), 10) == np.round(pitch, 10) and
            np.round(quat.yaw(), 10) == np.round(yaw, 10))

    axis = np.random.randint(1, 10, [3, ])
    angle = np.random.rand(1).item() * np.pi
    cAxis = NumCpp.NdArray(1, 3)
    cAxis.setArray(axis)
    assert np.array_equal(np.round(NumCpp.Quaternion.angleAxisRotationNdArray(cAxis, angle).flatten(), 10),
                          np.round(quatRotateAngleAxis(axis, angle), 10))

    axis = np.random.randint(1, 10, [3, ])
    angle = np.random.rand(1).item() * np.pi
    cAxis = NumCpp.NdArray(1, 3)
    cAxis.setArray(axis)
    assert np.array_equal(np.round(NumCpp.Quaternion.angleAxisRotationVec3(cAxis, angle).flatten(), 10),
                          np.round(quatRotateAngleAxis(axis, angle), 10))

    time = np.abs(np.random.randn(1) * 5).item()
    x1 = np.random.rand(3, 1).flatten()
    x1 = x1 / np.linalg.norm(x1)
    x2 = np.random.rand(3, 1).flatten()
    x2 = x2 / np.linalg.norm(x2)

    theta0 = np.random.rand(1).item() * 2 * np.pi
    theta = np.arccos(np.dot(x1, x2))
    theta1 = theta0 + theta
    cross = np.cross(x1, x2)
    cross = cross / np.linalg.norm(cross)
    cCross = NumCpp.NdArray(3, 1)
    cCross.setArray(np.reshape(cross, [3, 1]))

    q0 = np.array([cross[0] * np.sin(theta0 / 2),
                   cross[1] * np.sin(theta0 / 2),
                   cross[2] * np.sin(theta0 / 2),
                   np.cos(theta0 / 2)])
    q1 = np.array([cross[0] * np.sin(theta1 / 2),
                   cross[1] * np.sin(theta1 / 2),
                   cross[2] * np.sin(theta1 / 2),
                   np.cos(theta1 / 2)])
    quat0 = NumCpp.Quaternion(q0[0], q0[1], q0[2], q0[3])
    quat1 = NumCpp.Quaternion(q1[0], q1[1], q1[2], q1[3])
    crossTo = quat0.rotateNdArray(cCross).flatten()

    w = quat0.angularVelocity(quat1, time).flatten()
    angularVelocity = np.linalg.norm(w)
    axis = w / angularVelocity

    # round to 1 decimal place because C is an approximation on magnitude
    assert (np.round(angularVelocity * time - theta, 1) == 0 and
            np.all(np.round(axis, 9) == np.round(crossTo, 9)))

    quat = np.random.randint(1, 10, [4, ])
    unitQuat = quat / np.linalg.norm(quat)
    cArray = NumCpp.NdArray(1, 4)
    cArray.setArray(quat)
    quat = NumCpp.Quaternion(cArray)
    conjQuat = quat.conjugate()
    assert (np.round(conjQuat.i(), 10) == np.round(-unitQuat[0].item(), 10) and
            np.round(conjQuat.j(), 10) == np.round(-unitQuat[1].item(), 10) and
            np.round(conjQuat.k(), 10) == np.round(-unitQuat[2].item(), 10) and
            np.round(conjQuat.s(), 10) == np.round(unitQuat[3].item(), 10))

    quat = np.random.randint(1, 5, [4, ]).astype(np.double)
    cQuat = NumCpp.Quaternion(quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item())
    dcm = cQuat.toDCM()
    cArray = NumCpp.NdArray(3)
    cArray.setArray(dcm)
    assert np.array_equal(np.round(NumCpp.Quaternion(cArray).toNdArray().getNumpyArray().flatten(), 10),
                          np.round(quat / quatNorm(quat), 10))

    quat = NumCpp.Quaternion.identity()
    assert (quat.i() == 0 and
            quat.j() == 0 and
            quat.k() == 0 and
            quat.s() == 1)

    quat = np.random.randint(1, 10, [4, ])
    unitQuat = quat / np.linalg.norm(quat)
    cArray = NumCpp.NdArray(1, 4)
    cArray.setArray(quat)
    quat = NumCpp.Quaternion(cArray)
    conjQuat = quat.inverse()
    assert (np.round(conjQuat.i(), 10) == np.round(-unitQuat[0].item(), 10) and
            np.round(conjQuat.j(), 10) == np.round(-unitQuat[1].item(), 10) and
            np.round(conjQuat.k(), 10) == np.round(-unitQuat[2].item(), 10) and
            np.round(conjQuat.s(), 10) == np.round(unitQuat[3].item(), 10))

    myQuat1 = np.random.randint(1, 5, [4, ]).astype(np.double)
    myQuat2 = np.random.randint(1, 5, [4, ]).astype(np.double)
    cQuat1 = NumCpp.Quaternion(myQuat1[0].item(), myQuat1[1].item(), myQuat1[2].item(), myQuat1[3].item())
    cQuat2 = NumCpp.Quaternion(myQuat2[0].item(), myQuat2[1].item(), myQuat2[2].item(), myQuat2[3].item())
    t = np.random.rand(1).item()
    interpQuat = cQuat1.nlerp(cQuat2, t).flatten()
    assert np.array_equal(np.round(interpQuat, 10), np.round(nlerp(myQuat1, myQuat2, t), 10))

    cQuat1.print()

    myQuat = np.random.randint(1, 5, [4, ]).astype(np.double)
    cQuat = NumCpp.Quaternion(myQuat[0].item(), myQuat[1].item(), myQuat[2].item(), myQuat[3].item())
    vec = np.random.rand(3, 1) * 10
    cVec = NumCpp.NdArray(3, 1)
    cVec.setArray(vec)
    newVec = cQuat.rotateNdArray(cVec)
    newVecPy = np.array(cQuat.toDCM()).dot(np.array(vec))
    assert np.array_equal(np.round(newVec.flatten(), 10), np.round(newVecPy.flatten(), 10))

    myQuat = np.random.randint(1, 5, [4, ]).astype(np.double)
    cQuat = NumCpp.Quaternion(myQuat[0].item(), myQuat[1].item(), myQuat[2].item(), myQuat[3].item())
    vec = np.random.rand(3, 1) * 10
    cVec = NumCpp.NdArray(3, 1)
    cVec.setArray(vec)
    newVec = cQuat.rotateVec3(cVec)
    newVecPy = np.array(cQuat.toDCM()).dot(np.array(vec))
    assert np.array_equal(np.round(newVec.flatten(), 10), np.round(newVecPy.flatten(), 10))

    myQuat1 = np.random.randint(1, 5, [4, ]).astype(np.double)
    myQuat2 = np.random.randint(1, 5, [4, ]).astype(np.double)
    cQuat1 = NumCpp.Quaternion(myQuat1[0].item(), myQuat1[1].item(), myQuat1[2].item(), myQuat1[3].item())
    cQuat2 = NumCpp.Quaternion(myQuat2[0].item(), myQuat2[1].item(), myQuat2[2].item(), myQuat2[3].item())
    t = np.random.rand(1).item()
    interpQuatSlerp = cQuat1.slerp(cQuat2, t).flatten()
    interpQuatNlerp = cQuat1.nlerp(cQuat2, t).flatten()
    assert np.array_equal(np.round(interpQuatSlerp, 1), np.round(interpQuatNlerp, 1))

    quat = np.random.randint(1, 5, [4, ]).astype(np.double)
    unitQuat = quat / quatNorm(quat)
    cQuat = NumCpp.Quaternion(quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item())
    dcmPy = quat2dcm(unitQuat)
    dcm = cQuat.toDCM()
    assert np.array_equal(np.round(dcm, 10), np.round(dcmPy, 10))

    radians = np.random.rand(1) * 2 * np.pi
    quat = NumCpp.Quaternion.xRotation(radians.item()).toNdArray().getNumpyArray().flatten()
    assert np.array_equal(np.round(quat, 10), np.round(quatRotateX(radians.item()), 10))

    radians = np.random.rand(1) * 2 * np.pi
    quat = NumCpp.Quaternion.yRotation(radians.item()).toNdArray().getNumpyArray().flatten()
    assert np.array_equal(np.round(quat, 10), np.round(quatRotateY(radians.item()), 10))

    radians = np.random.rand(1) * 2 * np.pi
    quat = NumCpp.Quaternion.zRotation(radians.item()).toNdArray().getNumpyArray().flatten()
    assert np.array_equal(np.round(quat, 10), np.round(quatRotateZ(radians.item()), 10))

    quat1 = np.random.randint(1, 5, [4, ]).astype(np.double)
    cQuat1 = NumCpp.Quaternion(quat1[0].item(), quat1[1].item(), quat1[2].item(), quat1[3].item())
    cQuat2 = NumCpp.Quaternion(quat1[0].item(), quat1[1].item(), quat1[2].item(), quat1[3].item())
    assert cQuat1 == cQuat2

    quat1 = np.random.randint(1, 5, [4, ]).astype(np.double)
    quat2 = np.random.randint(1, 5, [4, ]).astype(np.double)
    cQuat1 = NumCpp.Quaternion(quat1[0].item(), quat1[1].item(), quat1[2].item(), quat1[3].item())
    cQuat2 = NumCpp.Quaternion(quat2[0].item(), quat2[1].item(), quat2[2].item(), quat2[3].item())
    assert cQuat1 != cQuat2

    quat1 = np.random.randint(1, 5, [4, ]).astype(np.double)
    quat2 = np.random.randint(1, 5, [4, ]).astype(np.double)
    cQuat1 = NumCpp.Quaternion(quat1[0].item(), quat1[1].item(), quat1[2].item(), quat1[3].item())
    cQuat2 = NumCpp.Quaternion(quat2[0].item(), quat2[1].item(), quat2[2].item(), quat2[3].item())
    resPy = quatAdd(quat1, quat2)
    res = cQuat1 + cQuat2
    assert np.array_equal(np.round(res.toNdArray().getNumpyArray().flatten(), 10), np.round(resPy, 10))

    quat1 = np.random.randint(1, 5, [4, ]).astype(np.double)
    quat2 = np.random.randint(1, 5, [4, ]).astype(np.double)
    cQuat1 = NumCpp.Quaternion(quat1[0].item(), quat1[1].item(), quat1[2].item(), quat1[3].item())
    cQuat2 = NumCpp.Quaternion(quat2[0].item(), quat2[1].item(), quat2[2].item(), quat2[3].item())
    resPy = quatSub(quat1, quat2)
    res = cQuat1 - cQuat2
    assert np.array_equal(np.round(res.flatten(), 10), np.round(resPy, 10))

    quat = np.random.randint(1, 5, [4, ]).astype(np.double)
    cQuat = NumCpp.Quaternion(quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item())
    assert np.array_equal(np.round((-cQuat).flatten(), 10), np.round(-quat/np.linalg.norm(quat), 10))

    quat = np.random.randint(1, 5, [4, ]).astype(np.double)
    cQuat = NumCpp.Quaternion(quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item())
    res = cQuat * -1
    assert np.array_equal(np.round(res.flatten(), 10), np.round(-quat / quatNorm(quat), 10))

    quat1 = np.random.randint(1, 5, [4, ]).astype(np.double)
    quat2 = np.random.randint(1, 5, [4, ]).astype(np.double)
    cQuat1 = NumCpp.Quaternion(quat1[0].item(), quat1[1].item(), quat1[2].item(), quat1[3].item())
    cQuat2 = NumCpp.Quaternion(quat2[0].item(), quat2[1].item(), quat2[2].item(), quat2[3].item())
    resPy = quatMult(quat1, quat2)
    res = cQuat1 * cQuat2
    assert np.array_equal(np.round(res.flatten(), 10), np.round(resPy, 10))

    quat = np.random.randint(1, 5, [4, ]).astype(np.double)
    cQuat = NumCpp.Quaternion(quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item())
    array = np.random.randint(1, 5, [3, 1])
    cArray = NumCpp.NdArray(3, 1)
    cArray.setArray(array)
    res = cQuat * cArray
    assert np.array_equal(np.round(res.flatten(), 10), np.round(np.dot(cQuat.toDCM(), array).flatten(), 10))

    quat1 = np.random.randint(1, 5, [4, ]).astype(np.double)
    quat2 = np.random.randint(1, 5, [4, ]).astype(np.double)
    cQuat1 = NumCpp.Quaternion(quat1[0].item(), quat1[1].item(), quat1[2].item(), quat1[3].item())
    cQuat2 = NumCpp.Quaternion(quat2[0].item(), quat2[1].item(), quat2[2].item(), quat2[3].item())
    resPy = quatDiv(quat1, quat2)
    res = cQuat1 / cQuat2
    assert np.array_equal(np.round(res.toNdArray().getNumpyArray().flatten(), 10), np.round(resPy, 10))


####################################################################################
def test_dcm():
    radians = np.random.rand(1) * 2 * np.pi
    axis = np.random.rand(3)
    cAxis = NumCpp.NdArray(1, 3)
    cAxis.setArray(axis)
    rot = NumCpp.DCM.angleAxisRotationNdArray(cAxis, radians.item())
    assert np.all(np.round(rot, 10) == np.round(angleAxisRotation(axis, radians.item()), 10))

    radians = np.random.rand(1) * 2 * np.pi
    axis = np.random.rand(3)
    cAxis = NumCpp.NdArray(1, 3)
    cAxis.setArray(axis)
    rot = NumCpp.DCM.angleAxisRotationVec3(cAxis, radians.item())
    assert np.all(np.round(rot, 10) == np.round(angleAxisRotation(axis, radians.item()), 10))

    radians = np.random.rand(1) * 2 * np.pi
    rot = NumCpp.DCM.xRotation(radians.item()).getNumpyArray()
    cArray = NumCpp.NdArray(3)
    cArray.setArray(rot)
    assert NumCpp.DCM.isValid(cArray)

    radians = np.random.rand(1) * 2 * np.pi
    rot = NumCpp.DCM.xRotation(radians.item()).getNumpyArray()
    assert np.all(np.round(rot, 10) == np.round(rotateX(radians.item()), 10))

    radians = np.random.rand(1) * 2 * np.pi
    rot = NumCpp.DCM.yRotation(radians.item()).getNumpyArray()
    assert np.all(np.round(rot, 10) == np.round(rotateY(radians.item()), 10))

    radians = np.random.rand(1) * 2 * np.pi
    rot = NumCpp.DCM.zRotation(radians.item()).getNumpyArray()
    assert np.all(np.round(rot, 10) == np.round(rotateZ(radians.item()), 10))


####################################################################################
def test_functions():
    k = np.random.randint(1, 5, [3, ]).astype(np.double)
    v = np.random.randint(1, 5, [3, ]).astype(np.double)
    theta = np.random.rand(1).item() * np.pi * 2
    vec = NumCpp.rodriguesRotation(k, theta, v).flatten()

    dcm = angleAxisRotation(k, theta)
    vecPy = dcm.dot(v).flatten()

    assert np.array_equal(np.round(vec, 10), np.round(vecPy, 10))

    radians = np.random.rand(1) * 2 * np.pi
    axis = np.random.rand(3)
    cAxis = NumCpp.NdArray(1, 3)
    cAxis.setArray(axis)
    rot = NumCpp.DCM.angleAxisRotationNdArray(cAxis, radians.item())

    vecBody = list()
    vecInertial = list()
    for _ in range(1000):
        vec = np.random.randint(1, 100, [3, ])
        vec = vec / np.linalg.norm(vec)
        vecBody.append(vec.flatten())

        vecInertial.append(rot.dot(vec).flatten())

    vecBody = np.array(vecBody)
    vecInertial = np.array(vecInertial)

    rotWahba = NumCpp.wahbasProblem(vecInertial, vecBody)

    assert np.array_equal(np.round(rotWahba, 10), np.round(rot, 10))

    radians = np.random.rand(1) * 2 * np.pi
    axis = np.random.rand(3)
    cAxis = NumCpp.NdArray(1, 3)
    cAxis.setArray(axis)
    rot = NumCpp.DCM.angleAxisRotationNdArray(cAxis, radians.item())

    vecBody = list()
    vecInertial = list()
    for _ in range(1000):
        vec = np.random.randint(1, 100, [3, ])
        vec = vec / np.linalg.norm(vec)
        vecBody.append(vec.flatten())

        vecInertial.append(rot.dot(vec).flatten())

    vecBody = np.array(vecBody)
    vecInertial = np.array(vecInertial)

    weights = np.random.randint(1, 100) * np.ones([vecBody.shape[0]])  # all the same weight for simplicity...

    rotWahba = NumCpp.wahbasProblemWeighted(vecInertial, vecBody, weights)

    assert np.array_equal(np.round(rotWahba, 10), np.round(rot, 10))


########################################################################################################################
def quatNorm(quat):
    return np.linalg.norm(quat)


########################################################################################################################
def dcm2quat(dcm):
    # http://www.vectornav.com/docs/default-source/documentation/vn-100-documentation/AN002.pdf?sfvrsn=19ee6b9_10

    q3 = 0.5 * np.sqrt(dcm[0, 0] + dcm[1, 1] + dcm[2, 2] + 1)
    q0 = (dcm[1, 2] - dcm[2, 1]) / (4 * q3)
    q1 = (dcm[2, 0] - dcm[0, 2]) / (4 * q3)
    q2 = (dcm[0, 1] - dcm[1, 0]) / (4 * q3)

    quat = np.array([q0, q1, q2, q3])
    quat = quat / np.linalg.norm(quat)

    return quat


########################################################################################################################
def quat2dcm(quat):
    # http://www.vectornav.com/docs/default-source/documentation/vn-100-documentation/AN002.pdf?sfvrsn=19ee6b9_10

    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]

    dcm = np.zeros([3, 3])
    dcm[0, 0] = q3**2 + q0**2 - q1**2 - q2**2
    dcm[0, 1] = 2 * (q0 * q1 - q3 * q2)
    dcm[0, 2] = 2 * (q0 * q2 + q3 * q1)
    dcm[1, 0] = 2 * (q0 * q1 + q3 * q2)
    dcm[1, 1] = q3**2 - q0**2 + q1**2 - q2**2
    dcm[1, 2] = 2 * (q1 * q2 - q3 * q0)
    dcm[2, 0] = 2 * (q0 * q2 - q3 * q1)
    dcm[2, 1] = 2 * (q1 * q2 + q3 * q0)
    dcm[2, 2] = q3**2 - q0**2 - q1**2 + q2**2

    return dcm


########################################################################################################################
def quatAdd(quat1, quat2):
    quat = quat1 / quatNorm(quat1) + quat2 / quatNorm(quat2)
    return quat / np.linalg.norm(quat)


########################################################################################################################
def quatSub(quat1, quat2):
    quat = quat1 / quatNorm(quat1) - quat2 / quatNorm(quat2)
    return quat / np.linalg.norm(quat)


########################################################################################################################
def quatMult(quat1, quat2):
    quat1N = quat1 / np.linalg.norm(quat1)
    quat2N = quat2 / np.linalg.norm(quat2)

    q0 = quat2N[3] * quat1N[0] + quat2N[0] * quat1N[3] - quat2N[1] * quat1N[2] + quat2N[2] * quat1N[1]
    q1 = quat2N[3] * quat1N[1] + quat2N[0] * quat1N[2] + quat2N[1] * quat1N[3] - quat2N[2] * quat1N[0]
    q2 = quat2N[3] * quat1N[2] - quat2N[0] * quat1N[1] + quat2N[1] * quat1N[0] + quat2N[2] * quat1N[3]
    q3 = quat2N[3] * quat1N[3] - quat2N[0] * quat1N[0] - quat2N[1] * quat1N[1] - quat2N[2] * quat1N[2]

    quat = np.array([q0, q1, q2, q3])
    return quat / np.linalg.norm(quat)


########################################################################################################################
def quatDiv(quat1, quat2):
    quat2Inv = -quat2
    quat2Inv[-1] *= -1
    return quatMult(quat1, quat2Inv)


########################################################################################################################
def nlerp(quat1, quat2, inT):
    quat1 = quat1 / np.linalg.norm(quat1)
    quat2 = quat2 / np.linalg.norm(quat2)

    oneMinusT = 1 - inT

    outQuat = np.zeros([4, ])

    outQuat[0] = oneMinusT * quat1[0] + inT * quat2[0]
    outQuat[1] = oneMinusT * quat1[1] + inT * quat2[1]
    outQuat[2] = oneMinusT * quat1[2] + inT * quat2[2]
    outQuat[3] = oneMinusT * quat1[3] + inT * quat2[3]

    outQuat = outQuat / np.linalg.norm(outQuat)

    return outQuat


########################################################################################################################
def quatRotateAngleAxis(axis, radians):
    axis = np.array(axis) / np.linalg.norm(axis)
    halfRadians = radians / 2
    quatList = [axis[0] * np.sin(halfRadians),
                axis[1] * np.sin(halfRadians),
                axis[2] * np.sin(halfRadians),
                np.cos(halfRadians)]
    return np.array(quatList) / np.linalg.norm(quatList)


########################################################################################################################
def quatRotateX(radians):
    return quatRotateAngleAxis([1, 0, 0], radians)


########################################################################################################################
def quatRotateY(radians):
    return quatRotateAngleAxis([0, 1, 0], radians)


########################################################################################################################
def quatRotateZ(radians):
    return quatRotateAngleAxis([0, 0, 1], radians)


########################################################################################################################
def angleAxisRotation(axis, radians):
    return quat2dcm(quatRotateAngleAxis(axis, radians))


########################################################################################################################
def rotateX(radians):
    return np.array([[1, 0, 0],
                     [0, np.cos(radians), -np.sin(radians)],
                     [0, np.sin(radians), np.cos(radians)]])


########################################################################################################################
def rotateY(radians):
    return np.array([[np.cos(radians), 0, np.sin(radians)],
                     [0, 1, 0],
                     [-np.sin(radians), 0, np.cos(radians)]])


########################################################################################################################
def rotateZ(radians):
    return np.array([[np.cos(radians), -np.sin(radians), 0],
                     [np.sin(radians), np.cos(radians), 0],
                     [0, 0, 1]])


########################################################################################################################
def hat(xyz):
    return np.array([[0, -xyz[2], xyz[1]],
                     [xyz[2], 0, -xyz[0]],
                     [-xyz[1], xyz[0], 0]])
