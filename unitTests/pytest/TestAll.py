import test_constants
import TestCoordinates
import TestDataCube
import test_dtypeInfo
import TestFilters
import TestFunctions
import TestImageProcessing
import TestIntegrate
import TestLinalg
import TestNdArray
import TestPolynomial
import TestRandom
import TestRotations
import TestRoots
import test_shape
import test_slice
import TestSpecial
import test_timer
import TestUtils
import TestVector


#################################################################################
def doTest():
    test_constants.doTest()
    TestCoordinates.doTest()
    TestDataCube.doTest()
    test_dtypeInfo.doTest()
    TestFilters.doTest()
    TestFunctions.doTest()
    TestImageProcessing.doTest()
    TestIntegrate.doTest()
    TestLinalg.doTest()
    TestNdArray.doTest()
    TestPolynomial.doTest()
    TestRandom.doTest()
    TestRotations.doTest()
    TestRoots.doTest()
    test_shape.doTest()
    test_slice.doTest()
    TestSpecial.doTest()
    test_timer.doTest()
    TestUtils.doTest()
    TestVector.doTest()


#################################################################################
if __name__ == '__main__':
    doTest()
