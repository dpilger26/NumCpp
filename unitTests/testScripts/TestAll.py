import TestConstants
import TestCoordinates
import TestDataCube
import TestDtypeInfo
import TestFilters
import TestFunctions
import TestImageProcessing
import TestLinalg
import TestNdArray
import TestPolynomial
import TestRandom
import TestRotations
import TestShape
import TestSlice
import TestSpecial
import TestTimer
import TestUtils
import TestVector


#################################################################################
def doTest():
    TestConstants.doTest()
    TestCoordinates.doTest()
    TestDataCube.doTest()
    TestDtypeInfo.doTest()
    TestFilters.doTest()
    TestFunctions.doTest()
    TestImageProcessing.doTest()
    TestLinalg.doTest()
    TestNdArray.doTest()
    TestPolynomial.doTest()
    TestRandom.doTest()
    TestRotations.doTest()
    TestShape.doTest()
    TestSlice.doTest()
    TestSpecial.doTest()
    TestTimer.doTest()
    TestUtils.doTest()
    TestVector.doTest()


#################################################################################
if __name__ == '__main__':
    doTest()
