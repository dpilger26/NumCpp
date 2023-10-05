#include "NumCpp/Random.hpp"

#include "BindingsIncludes.hpp"

//================================================================================

namespace RandomInterface
{
    template<typename dtype>
    dtype choiceSingle(const NdArray<dtype>& inArray)
    {
        return random::choice(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric choiceMultiple(const NdArray<dtype>& inArray, uint32 inNum, Replace replace)
    {
        return nc2pybind(random::choice(inArray, inNum, replace));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric permutationScalar(dtype inValue)
    {
        return nc2pybind(random::permutation(inValue));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric permutationArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(random::permutation(inArray));
    }

    namespace RNG
    {
        template<typename RNG_t>
        bool bernoulliValue(RNG_t rng, double inP)
        {
            return rng.bernoulli(inP);
        }

        //============================================================================

        template<typename RNG_t>
        pbArrayGeneric bernoulliShape(RNG_t rng, const Shape& inShape, double inP)
        {
            return nc2pybind(rng.bernoulli(inShape, inP));
        }

        //============================================================================

#ifndef NUMCPP_NO_USE_BOOST
        template<typename RNG_t, typename dtype>
        dtype betaValue(RNG_t rng, dtype inAlpha, dtype inBeta)
        {
            return rng.beta(inAlpha, inBeta);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric betaShape(RNG_t rng, const Shape& inShape, dtype inAlpha, dtype inBeta)
        {
            return nc2pybind(rng.beta(inShape, inAlpha, inBeta));
        }
#endif

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype binomialValue(RNG_t rng, dtype inN, double inP)
        {
            return rng.binomial(inN, inP);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric binomialShape(RNG_t rng, const Shape& inShape, dtype inN, double inP)
        {
            return nc2pybind(rng.binomial(inShape, inN, inP));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype cauchyValue(RNG_t rng, dtype inMean, dtype inSigma)
        {
            return rng.cauchy(inMean, inSigma);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric cauchyShape(RNG_t rng, const Shape& inShape, dtype inMean, dtype inSigma)
        {
            return nc2pybind(rng.cauchy(inShape, inMean, inSigma));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype chiSquareValue(RNG_t rng, dtype inDof)
        {
            return rng.chiSquare(inDof);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric chiSquareShape(RNG_t rng, const Shape& inShape, dtype inDof)
        {
            return nc2pybind(rng.chiSquare(inShape, inDof));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype choiceValue(RNG_t rng, const NdArray<dtype>& inArray)
        {
            return rng.choice(inArray);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric choiceShape(RNG_t rng, const NdArray<dtype>& inArray, uint32 inNum, Replace replace)
        {
            return nc2pybind(rng.choice(inArray, inNum, replace));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype discreteValue(RNG_t rng, const NdArray<double>& inWeights)
        {
            return rng.template discrete<dtype>(inWeights);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric discreteShape(RNG_t rng, const Shape& inShape, const NdArray<double>& inWeights)
        {
            return nc2pybind(rng.template discrete<dtype>(inShape, inWeights));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype exponentialValue(RNG_t rng, dtype inScaleValue)
        {
            return rng.exponential(inScaleValue);
        }

        template<typename RNG_t, typename dtype>
        pbArrayGeneric exponentialShape(RNG_t rng, const Shape& inShape, dtype inScaleValue)
        {
            return nc2pybind(rng.exponential(inShape, inScaleValue));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype extremeValueValue(RNG_t rng, dtype inA = 1, dtype inB = 1)
        {
            return rng.extremeValue(inA, inB);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric extremeValueShape(RNG_t rng, const Shape& inShape, dtype inA, dtype inB)
        {
            return nc2pybind(rng.extremeValue(inShape, inA, inB));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype fValue(RNG_t rng, dtype inDofN, dtype inDofD)
        {
            return rng.f(inDofN, inDofD);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric fShape(RNG_t rng, const Shape& inShape, dtype inDofN, dtype inDofD)
        {
            return nc2pybind(rng.f(inShape, inDofN, inDofD));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype gammaValue(RNG_t rng, dtype inGammaShape, dtype inScaleValue)
        {
            return rng.gamma(inGammaShape, inScaleValue);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric gammaShape(RNG_t rng, const Shape& inShape, dtype inGammaShape, dtype inScaleValue)
        {
            return nc2pybind(rng.gamma(inShape, inGammaShape, inScaleValue));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype geometricValue(RNG_t rng, double inP)
        {
            return rng.template geometric<dtype>(inP);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric geometricShape(RNG_t rng, const Shape& inShape, double inP)
        {
            return nc2pybind(rng.template geometric<dtype>(inShape, inP));
        }

        //============================================================================

#ifndef NUMCPP_NO_USE_BOOST
        template<typename RNG_t, typename dtype>
        dtype laplaceValue(RNG_t rng, dtype inLoc, dtype inScale)
        {
            return rng.laplace(inLoc, inScale);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric laplaceShape(RNG_t rng, const Shape& inShape, dtype inLoc, dtype inScale)
        {
            return nc2pybind(rng.laplace(inShape, inLoc, inScale));
        }
#endif

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype lognormalValue(RNG_t rng, dtype inMean, dtype inSigma)
        {
            return rng.lognormal(inMean, inSigma);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric lognormalShape(RNG_t rng, const Shape& inShape, dtype inMean, dtype inSigma)
        {
            return nc2pybind(rng.lognormal(inShape, inMean, inSigma));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype negativeBinomialValue(RNG_t rng, dtype inN, double inP = 0.5)
        {
            return rng.negativeBinomial(inN, inP);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric negativeBinomialShape(RNG_t rng, const Shape& inShape, dtype inN, double inP)
        {
            return nc2pybind(rng.negativeBinomial(inShape, inN, inP));
        }

        //============================================================================

#ifndef NUMCPP_NO_USE_BOOST
        template<typename RNG_t, typename dtype>
        dtype nonCentralChiSquaredValue(RNG_t rng, dtype inK, dtype inLambda)
        {
            return rng.nonCentralChiSquared(inK, inLambda);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric nonCentralChiSquaredShape(RNG_t rng, const Shape& inShape, dtype inK, dtype inLambda)
        {
            return nc2pybind(rng.nonCentralChiSquared(inShape, inK, inLambda));
        }
#endif

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype normalValue(RNG_t rng, dtype inMean, dtype inSigma)
        {
            return rng.normal(inMean, inSigma);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric normalShape(RNG_t rng, const Shape& inShape, dtype inMean, dtype inSigma)
        {
            return nc2pybind(rng.normal(inShape, inMean, inSigma));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric permutationValue(RNG_t rng, dtype inValue)
        {
            return nc2pybind(rng.permutation(inValue));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric permutationShape(RNG_t rng, const NdArray<dtype>& inArray)
        {
            return nc2pybind(rng.permutation(inArray));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype poissonValue(RNG_t rng, double mean)
        {
            return rng.template poisson<dtype>(mean);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric poissonShape(RNG_t rng, const Shape& inShape, double mean)
        {
            return nc2pybind(rng.template poisson<dtype>(inShape, mean));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype randValue(RNG_t rng)
        {
            return rng.template rand<dtype>();
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric randShape(RNG_t rng, const Shape& inShape)
        {
            return nc2pybind(rng.template rand<dtype>(inShape));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype randFloatValue(RNG_t rng, dtype inLow, dtype inHigh)
        {
            return rng.randFloat(inLow, inHigh);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric randFloatShape(RNG_t rng, const Shape& inShape, dtype inLow, dtype inHigh)
        {
            return nc2pybind(rng.randFloat(inShape, inLow, inHigh));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype randIntValue(RNG_t rng, dtype inLow, dtype inHigh)
        {
            return rng.randInt(inLow, inHigh);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric randIntShape(RNG_t rng, const Shape& inShape, dtype inLow, dtype inHigh)
        {
            return nc2pybind(rng.randInt(inShape, inLow, inHigh));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype randNValue(RNG_t rng)
        {
            return rng.template randN<dtype>();
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric randNShape(RNG_t rng, const Shape& inShape)
        {
            return nc2pybind(rng.template randN<dtype>(inShape));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype standardNormalValue(RNG_t rng)
        {
            return rng.template standardNormal<dtype>();
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric standardNormalShape(RNG_t rng, const Shape& inShape)
        {
            return nc2pybind(rng.template standardNormal<dtype>(inShape));
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype studentTValue(RNG_t rng, dtype inDof)
        {
            return rng.studentT(inDof);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric studentTShape(RNG_t rng, const Shape& inShape, dtype inDof)
        {
            return nc2pybind(rng.studentT(inShape, inDof));
        }

        //============================================================================

#ifndef NUMCPP_NO_USE_BOOST
        template<typename RNG_t, typename dtype>
        dtype triangleValue(RNG_t rng, dtype inA, dtype inB, dtype inC)
        {
            return rng.triangle(inA, inB, inC);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric triangleShape(RNG_t rng, const Shape& inShape, dtype inA, dtype inB, dtype inC)
        {
            return nc2pybind(rng.triangle(inShape, inA, inB, inC));
        }
#endif

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype uniformValue(RNG_t rng, dtype inLow, dtype inHigh)
        {
            return rng.uniform(inLow, inHigh);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric uniformShape(RNG_t rng, const Shape& inShape, dtype inLow, dtype inHigh)
        {
            return nc2pybind(rng.uniform(inShape, inLow, inHigh));
        }

        //============================================================================

#ifndef NUMCPP_NO_USE_BOOST
        template<typename RNG_t, typename dtype>
        pbArrayGeneric uniformOnSphere(RNG_t rng, uint32 inNumPoints, uint32 inDims)
        {
            return nc2pybind(rng.template uniformOnSphere<dtype>(inNumPoints, inDims));
        }
#endif

        //============================================================================

        template<typename RNG_t, typename dtype>
        dtype weibullValue(RNG_t rng, dtype inA, dtype inB)
        {
            return rng.weibull(inA, inB);
        }

        //============================================================================

        template<typename RNG_t, typename dtype>
        pbArrayGeneric weibullShape(RNG_t rng, const Shape& inShape, dtype inA, dtype inB)
        {
            return nc2pybind(rng.weibull(inShape, inA, inB));
        }
    } // namespace RNG
} // namespace RandomInterface

//================================================================================

void initRandom(pb11::module& m)
{
    // Random.hpp
    NdArray<bool> (*bernoulliArray)(const Shape&, double) = &random::bernoulli;
    bool (*bernoilliScalar)(double)                       = &random::bernoulli;
    m.def("bernoulli", bernoulliArray);
    m.def("bernoulli", bernoilliScalar);

#ifndef NUMCPP_NO_USE_BOOST
    NdArray<double> (*betaArray)(const Shape&, double, double) = &random::beta<double>;
    double (*betaScalar)(double, double)                       = &random::beta<double>;
    m.def("beta", betaArray);
    m.def("beta", betaScalar);
#endif

    NdArray<int32> (*binomialArray)(const Shape&, int32, double) = &random::binomial<int32>;
    int32 (*binomialScalar)(int32, double)                       = &random::binomial<int32>;
    m.def("binomial", binomialArray);
    m.def("binomial", binomialScalar);

    NdArray<double> (*cauchyArray)(const Shape&, double, double) = &random::cauchy<double>;
    double (*cauchyScalar)(double, double)                       = &random::cauchy<double>;
    m.def("cauchy", cauchyArray);
    m.def("cauchy", cauchyScalar);

    NdArray<double> (*chiSquareArray)(const Shape&, double) = &random::chiSquare<double>;
    double (*chiSquareScalar)(double)                       = &random::chiSquare<double>;
    m.def("chiSquare", chiSquareArray);
    m.def("chiSquare", chiSquareScalar);

    m.def("choiceSingle", &RandomInterface::choiceSingle<double>);
    m.def("choiceMultiple", &RandomInterface::choiceMultiple<double>);

    NdArray<int32> (*discreteArray)(const Shape&, const NdArray<double>&) = &random::discrete<int32>;
    int32 (*discreteScalar)(const NdArray<double>&)                       = &random::discrete<int32>;
    m.def("discrete", discreteArray);
    m.def("discrete", discreteScalar);

    NdArray<double> (*exponentialArray)(const Shape&, double) = &random::exponential<double>;
    double (*exponentialScalar)(double)                       = &random::exponential<double>;
    m.def("exponential", exponentialArray);
    m.def("exponential", exponentialScalar);

    NdArray<double> (*extremeValueArray)(const Shape&, double, double) = &random::extremeValue<double>;
    double (*extremeValueScalar)(double, double)                       = &random::extremeValue<double>;
    m.def("extremeValue", extremeValueArray);
    m.def("extremeValue", extremeValueScalar);

    NdArray<double> (*fArray)(const Shape&, double, double) = &random::f<double>;
    double (*fScalar)(double, double)                       = &random::f<double>;
    m.def("f", fArray);
    m.def("f", fScalar);

    NdArray<double> (*gammaArray)(const Shape&, double, double) = &random::gamma<double>;
    double (*gammaScalar)(double, double)                       = &random::gamma<double>;
    m.def("gamma", gammaArray);
    m.def("gamma", gammaScalar);

    NdArray<int32> (*geometricArray)(const Shape&, double) = &random::geometric<int32>;
    int32 (*geometricScalar)(double)                       = &random::geometric<int32>;
    m.def("geometric", geometricArray);
    m.def("geometric", geometricScalar);

#ifndef NUMCPP_NO_USE_BOOST
    NdArray<double> (*laplaceArray)(const Shape&, double, double) = &random::laplace<double>;
    double (*laplaceScalar)(double, double)                       = &random::laplace<double>;
    m.def("laplace", laplaceArray);
    m.def("laplace", laplaceScalar);
#endif

    NdArray<double> (*lognormalArray)(const Shape&, double, double) = &random::lognormal<double>;
    double (*lognormalScalar)(double, double)                       = &random::lognormal<double>;
    m.def("lognormal", lognormalArray);
    m.def("lognormal", lognormalScalar);

    NdArray<int32> (*negativeBinomialArray)(const Shape&, int32, double) = &random::negativeBinomial<int32>;
    int32 (*negativeBinomialScalar)(int32, double)                       = &random::negativeBinomial<int32>;
    m.def("negativeBinomial", negativeBinomialArray);
    m.def("negativeBinomial", negativeBinomialScalar);

#ifndef NUMCPP_NO_USE_BOOST
    NdArray<double> (*nonCentralChiSquaredArray)(const Shape&, double, double) = &random::nonCentralChiSquared<double>;
    double (*nonCentralChiSquaredScalar)(double, double)                       = &random::nonCentralChiSquared<double>;
    m.def("nonCentralChiSquared", nonCentralChiSquaredArray);
    m.def("nonCentralChiSquared", nonCentralChiSquaredScalar);
#endif

    NdArray<double> (*normalArray)(const Shape&, double, double) = &random::normal<double>;
    double (*normalScalar)(double, double)                       = &random::normal<double>;
    m.def("normal", normalArray);
    m.def("normal", normalScalar);

    m.def("permutationScalar", &RandomInterface::permutationScalar<double>);
    m.def("permutationArray", &RandomInterface::permutationArray<double>);

    NdArray<int32> (*poissonArray)(const Shape&, double) = &random::poisson<int32>;
    int32 (*poissonScalar)(double)                       = &random::poisson<int32>;
    m.def("poisson", poissonArray);
    m.def("poisson", poissonScalar);

    NdArray<double> (*randArray)(const Shape&) = &random::rand<double>;
    double (*randScalar)()                     = &random::rand<double>;
    m.def("rand", randArray);
    m.def("rand", randScalar);

    NdArray<double> (*randFloatArray)(const Shape&, double, double) = &random::randFloat<double>;
    double (*randFloatScalar)(double, double)                       = &random::randFloat<double>;
    m.def("randFloat", randFloatArray);
    m.def("randFloat", randFloatScalar);

    NdArray<int32> (*randIntArray)(const Shape&, int32, int32) = &random::randInt<int32>;
    int32 (*randIntScalar)(int32, int32)                       = &random::randInt<int32>;
    m.def("randInt", randIntArray);
    m.def("randInt", randIntScalar);

    NdArray<double> (*randNArray)(const Shape&) = &random::randN<double>;
    double (*randNScalar)()                     = &random::randN<double>;
    m.def("randN", randNArray);
    m.def("randN", randNScalar);

    m.def("seed", &random::seed);
    m.def("shuffle", &random::shuffle<double>);

    NdArray<double> (*standardNormalArray)(const Shape&) = &random::standardNormal<double>;
    double (*standardNormalScalar)()                     = &random::standardNormal<double>;
    m.def("standardNormal", standardNormalArray);
    m.def("standardNormal", standardNormalScalar);

    NdArray<double> (*studentTArray)(const Shape&, double) = &random::studentT<double>;
    double (*studentTScalar)(double)                       = &random::studentT<double>;
    m.def("studentT", studentTArray);
    m.def("studentT", studentTScalar);

#ifndef NUMCPP_NO_USE_BOOST
    NdArray<double> (*triangleArray)(const Shape&, double, double, double) = &random::triangle<double>;
    double (*triangleScalar)(double, double, double)                       = &random::triangle<double>;
    m.def("triangle", triangleArray);
    m.def("triangle", triangleScalar);
#endif

    NdArray<double> (*uniformArray)(const Shape&, double, double) = &random::uniform<double>;
    double (*uniformScalar)(double, double)                       = &random::uniform<double>;
    m.def("uniform", uniformArray);
    m.def("uniform", uniformScalar);

#ifndef NUMCPP_NO_USE_BOOST
    m.def("uniformOnSphere", &random::uniformOnSphere<double>);
#endif

    NdArray<double> (*weibullArray)(const Shape&, double, double) = &random::weibull<double>;
    double (*weibullScalar)(double, double)                       = &random::weibull<double>;
    m.def("weibull", weibullArray);
    m.def("weibull", weibullScalar);

    using RNG_t = random::RNG<>;
    pb11::class_<RNG_t>(m, "RNG")
        .def(pb11::init<>())
        .def(pb11::init<int>())
        .def("bernoulli", &RandomInterface::RNG::bernoulliValue<RNG_t>)
        .def("bernoulli", &RandomInterface::RNG::bernoulliShape<RNG_t>)
#ifndef NUMCPP_NO_USE_BOOST
        .def("beta", &RandomInterface::RNG::betaValue<RNG_t, double>)
        .def("beta", &RandomInterface::RNG::betaShape<RNG_t, double>)
#endif
        .def("binomial", &RandomInterface::RNG::binomialValue<RNG_t, int32>)
        .def("binomial", &RandomInterface::RNG::binomialShape<RNG_t, int32>)
        .def("cauchy", &RandomInterface::RNG::cauchyValue<RNG_t, double>)
        .def("cauchy", &RandomInterface::RNG::cauchyShape<RNG_t, double>)
        .def("chiSquare", &RandomInterface::RNG::chiSquareValue<RNG_t, double>)
        .def("chiSquare", &RandomInterface::RNG::chiSquareShape<RNG_t, double>)
        .def("choice", &RandomInterface::RNG::choiceValue<RNG_t, double>)
        .def("choice", &RandomInterface::RNG::choiceShape<RNG_t, double>)
        .def("discrete", &RandomInterface::RNG::discreteValue<RNG_t, int32>)
        .def("discrete", &RandomInterface::RNG::discreteShape<RNG_t, int32>)
        .def("exponential", &RandomInterface::RNG::exponentialValue<RNG_t, double>)
        .def("exponential", &RandomInterface::RNG::exponentialShape<RNG_t, double>)
        .def("extremeValue", &RandomInterface::RNG::extremeValueValue<RNG_t, double>)
        .def("extremeValue", &RandomInterface::RNG::extremeValueShape<RNG_t, double>)
        .def("f", &RandomInterface::RNG::fValue<RNG_t, double>)
        .def("f", &RandomInterface::RNG::fShape<RNG_t, double>)
        .def("gamma", &RandomInterface::RNG::gammaValue<RNG_t, double>)
        .def("gamma", &RandomInterface::RNG::gammaShape<RNG_t, double>)
        .def("geometric", &RandomInterface::RNG::geometricValue<RNG_t, int32>)
        .def("geometric", &RandomInterface::RNG::geometricShape<RNG_t, int32>)
#ifndef NUMCPP_NO_USE_BOOST
        .def("laplace", &RandomInterface::RNG::laplaceValue<RNG_t, double>)
        .def("laplace", &RandomInterface::RNG::laplaceShape<RNG_t, double>)
#endif
        .def("lognormal", &RandomInterface::RNG::lognormalValue<RNG_t, double>)
        .def("lognormal", &RandomInterface::RNG::lognormalShape<RNG_t, double>)
        .def("negativeBinomial", &RandomInterface::RNG::negativeBinomialValue<RNG_t, int32>)
        .def("negativeBinomial", &RandomInterface::RNG::negativeBinomialShape<RNG_t, int32>)
#ifndef NUMCPP_NO_USE_BOOST
        .def("nonCentralChiSquared", &RandomInterface::RNG::nonCentralChiSquaredValue<RNG_t, double>)
        .def("nonCentralChiSquared", &RandomInterface::RNG::nonCentralChiSquaredShape<RNG_t, double>)
#endif
        .def("normal", &RandomInterface::RNG::normalValue<RNG_t, double>)
        .def("normal", &RandomInterface::RNG::normalShape<RNG_t, double>)
        .def("permutation", &RandomInterface::RNG::permutationValue<RNG_t, double>)
        .def("permutation", &RandomInterface::RNG::permutationShape<RNG_t, double>)
        .def("poisson", &RandomInterface::RNG::poissonValue<RNG_t, int32>)
        .def("poisson", &RandomInterface::RNG::poissonShape<RNG_t, int32>)
        .def("rand", &RandomInterface::RNG::randValue<RNG_t, double>)
        .def("rand", &RandomInterface::RNG::randShape<RNG_t, double>)
        .def("randFloat", &RandomInterface::RNG::randFloatValue<RNG_t, double>)
        .def("randFloat", &RandomInterface::RNG::randFloatShape<RNG_t, double>)
        .def("randInt", &RandomInterface::RNG::randIntValue<RNG_t, int>)
        .def("randInt", &RandomInterface::RNG::randIntShape<RNG_t, int>)
        .def("randN", &RandomInterface::RNG::randNValue<RNG_t, double>)
        .def("randN", &RandomInterface::RNG::randNShape<RNG_t, double>)
        .def("seed", &RNG_t::seed)
        .def("shuffle", &RNG_t::shuffle<double>)
        .def("standardNormal", &RandomInterface::RNG::standardNormalValue<RNG_t, double>)
        .def("standardNormal", &RandomInterface::RNG::standardNormalShape<RNG_t, double>)
        .def("studentT", &RandomInterface::RNG::studentTValue<RNG_t, double>)
        .def("studentT", &RandomInterface::RNG::studentTShape<RNG_t, double>)
#ifndef NUMCPP_NO_USE_BOOST
        .def("triangle", &RandomInterface::RNG::triangleValue<RNG_t, double>)
        .def("triangle", &RandomInterface::RNG::triangleShape<RNG_t, double>)
#endif
        .def("uniform", &RandomInterface::RNG::uniformValue<RNG_t, double>)
        .def("uniform", &RandomInterface::RNG::uniformShape<RNG_t, double>)
#ifndef NUMCPP_NO_USE_BOOST
        .def("uniformOnSphere", &RandomInterface::RNG::uniformOnSphere<RNG_t, double>)
#endif
        .def("weibull", &RandomInterface::RNG::weibullValue<RNG_t, double>)
        .def("weibull", &RandomInterface::RNG::weibullShape<RNG_t, double>);
}