import TestDataCube
import TestShape
import TestSlice
import TestTimer
import TestNdArray
import TestMethods
import TestConstants
import TestCoordinates
import TestFilters
import TestImageProcessing
import TestLinalg
import TestRandom
import TestRotations
import TestPolynomial
import TestUtils
import TestVector
import TestDtypeInfo


#################################################################################
def doTest():
    TestDataCube.doTest()
    TestShape.doTest()
    TestSlice.doTest()
    TestTimer.doTest()
    TestUtils.doTest()
    TestDtypeInfo.doTest()
    TestNdArray.doTest()
    TestMethods.doTest()
    TestCoordinates.doTest()
    TestConstants.doTest()
    TestLinalg.doTest()
    TestRandom.doTest()
    TestRotations.doTest()
    TestFilters.doTest()
    TestPolynomial.doTest()
    TestVector.doTest()
    TestImageProcessing.doTest()


#################################################################################
if __name__ == '__main__':
    doTest()
