import numpy as np
from termcolor import colored
import sys
sys.path.append(r'../build/x64/Release')
import NumC

####################################################################################
def doTest():
    print(colored('Testing Rotations Module', 'magenta'))

    print(colored('Testing Default Constructor', 'cyan'))
    quat = NumC.Quaternion()
    print(colored('\tPASS', 'green'))

    print(colored('Testing Value Constructor', 'cyan'))
    quat = np.random.randint(1,10, [4,])
    quatNorm = quat / np.linalg.norm(quat)
    quat = NumC.Quaternion(quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item())
    if (quat.i() == quatNorm[0].item() and
        quat.j() == quatNorm[1].item() and
        quat.k() == quatNorm[2].item() and
        quat.s() == quatNorm[3].item()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Array Constructor', 'cyan'))
    quat = np.random.randint(1,10, [4,])
    quatNorm = quat / np.linalg.norm(quat)
    cArray = NumC.NdArray(1, 4)
    cArray.setArray(quat)
    quat = NumC.Quaternion(cArray)
    if (quat.i() == quatNorm[0].item() and
        quat.j() == quatNorm[1].item() and
        quat.k() == quatNorm[2].item() and
        quat.s() == quatNorm[3].item()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing angleAxisRotation', 'cyan'))
    axis = np.random.randint(1,10, [3,])
    angle = np.random.rand(1).item() * np.pi
    cAxis = NumC.NdArray(1, 3)
    cAxis.setArray(axis)
    if np.array_equal(np.round(NumC.Quaternion.angleAxisRotation(cAxis, angle).toNdArray().getNumpyArray().flatten(), 10),
                      np.round(angleAxisRotation(axis, angle), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing conjugate', 'cyan'))
    quat = np.random.randint(1,10, [4,])
    quatNorm = quat / np.linalg.norm(quat)
    cArray = NumC.NdArray(1, 4)
    cArray.setArray(quat)
    quat = NumC.Quaternion(cArray)
    conjQuat = quat.conjugate()
    if (np.round(conjQuat.i(), 10) == np.round(-quatNorm[0].item(), 10) and
        np.round(conjQuat.j(), 10) == np.round(-quatNorm[1].item(), 10) and
        np.round(conjQuat.k(), 10) == np.round(-quatNorm[2].item(), 10) and
        np.round(conjQuat.s(), 10) == np.round(quatNorm[3].item(), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing inverse', 'cyan'))
    quat = np.random.randint(1,10, [4,])
    quatNorm = quat / np.linalg.norm(quat)
    cArray = NumC.NdArray(1, 4)
    cArray.setArray(quat)
    quat = NumC.Quaternion(cArray)
    conjQuat = quat.inverse()
    if (np.round(conjQuat.i(), 10) == np.round(-quatNorm[0].item(), 10) and
        np.round(conjQuat.j(), 10) == np.round(-quatNorm[1].item(), 10) and
        np.round(conjQuat.k(), 10) == np.round(-quatNorm[2].item(), 10) and
        np.round(conjQuat.s(), 10) == np.round(quatNorm[3].item(), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

####################################################################################
def angleAxisRotation(inAxis, inAngle):
    normAxis = inAxis / np.linalg.norm(inAxis)

    quat = np.zeros([4,])
    quat[0] = normAxis[0] * np.sin(inAngle / 2)
    quat[1] = normAxis[1] * np.sin(inAngle / 2)
    quat[2] = normAxis[2] * np.sin(inAngle / 2)
    quat[3] = np.cos(inAngle / 2)

    quat = quat / np.linalg.norm(quat)

    return quat

####################################################################################
if __name__ == '__main__':
    doTest()