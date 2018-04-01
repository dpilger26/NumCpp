import TestShape
import TestSlice
import TestTimer
import TestNdArray
import TestMethods
import TestConstants
import TestImageProcessing
import TestLinalg
import TestRandom
import TestRotations
import TestPolynomial
import TestFFT
import TestUtils
import TestDtypeInfo

#################################################################################
def doTest():
    TestShape.doTest()
    TestSlice.doTest()
    TestTimer.doTest()
    TestUtils.doTest()
    TestDtypeInfo.doTest()
    TestNdArray.doTest()
    TestMethods.doTest()
    TestConstants.doTest()
    TestLinalg.doTest()
    TestRandom.doTest()
    TestRotations.doTest()
    TestPolynomial.doTest()
    TestFFT.doTest()
    TestImageProcessing.doTest()

#################################################################################
if __name__ == '__main__':
    doTest()