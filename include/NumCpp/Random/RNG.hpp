/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.1
///
/// License
/// Copyright 2018-2022 David Pilger
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy of this
/// software and associated documentation files(the "Software"), to deal in the Software
/// without restriction, including without limitation the rights to use, copy, modify,
/// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
/// permit persons to whom the Software is furnished to do so, subject to the following
/// conditions :
///
/// The above copyright notice and this permission notice shall be included in all copies
/// or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
/// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
/// PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
/// FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
/// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
/// DEALINGS IN THE SOFTWARE.
///
/// Description
/// Random Number Generater Class with non-global state
///
#pragma once

#include <random>

#include "NumCpp/Random/bernoulli.hpp"
#include "NumCpp/Random/beta.hpp"
#include "NumCpp/Random/binomial.hpp"
#include "NumCpp/Random/cauchy.hpp"
#include "NumCpp/Random/chiSquare.hpp"
#include "NumCpp/Random/choice.hpp"
#include "NumCpp/Random/discrete.hpp"
#include "NumCpp/Random/exponential.hpp"
#include "NumCpp/Random/extremeValue.hpp"
#include "NumCpp/Random/f.hpp"
#include "NumCpp/Random/gamma.hpp"
#include "NumCpp/Random/geometric.hpp"
#include "NumCpp/Random/laplace.hpp"
#include "NumCpp/Random/lognormal.hpp"
#include "NumCpp/Random/negativeBinomial.hpp"
#include "NumCpp/Random/nonCentralChiSquared.hpp"
#include "NumCpp/Random/normal.hpp"
#include "NumCpp/Random/permutation.hpp"
#include "NumCpp/Random/poisson.hpp"
#include "NumCpp/Random/rand.hpp"
#include "NumCpp/Random/randFloat.hpp"
#include "NumCpp/Random/randInt.hpp"
#include "NumCpp/Random/randN.hpp"
#include "NumCpp/Random/shuffle.hpp"
#include "NumCpp/Random/standardNormal.hpp"
#include "NumCpp/Random/studentT.hpp"
#include "NumCpp/Random/triangle.hpp"
#include "NumCpp/Random/uniform.hpp"
#include "NumCpp/Random/uniformOnSphere.hpp"
#include "NumCpp/Random/weibull.hpp"

namespace nc
{
    namespace random
    {
        //============================================================================
        // Class Description:
        /// Random Number Generater Class with non-global state
        ///
        template<typename GeneratorType = std::mt19937_64>
        class RNG
        {
        public:
            //============================================================================
            // Method Description:
            /// Defualt Constructor
            ///
            RNG() = default;

            //============================================================================
            // Method Description:
            /// Seed Constructor
            ///
            /// @param seed: the seed value
            ///
            RNG(int seed) :
                generator_(seed){};

            //============================================================================
            // Method Description:
            /// Single random value sampled from the "bernoulli" distribution.
            ///
            /// @param inP (probability of success [0, 1]). Default 0.5
            /// @return NdArray
            ///
            bool bernoulli(double inP = 0.5)
            {
                return detail::bernoulli(generator_, inP);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from the "bernoulli" distribution.
            ///
            /// @param inShape
            /// @param inP (probability of success [0, 1]). Default 0.5
            /// @return NdArray
            ///
            NdArray<bool> bernoulli(const Shape& inShape, double inP = 0.5)
            {
                return detail::bernoulli(generator_, inShape, inP);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the from the "beta" distribution.
            /// NOTE: Use of this function requires using the Boost includes.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.beta.html#numpy.random.beta
            ///
            /// @param inAlpha
            /// @param inBeta
            /// @return NdArray
            ///
            template<typename dtype>
            dtype beta(dtype inAlpha, dtype inBeta)
            {
                return detail::beta(generator_, inAlpha, inBeta);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from the "beta" distribution.
            /// NOTE: Use of this function requires using the Boost includes.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.beta.html#numpy.random.beta
            ///
            /// @param inShape
            /// @param inAlpha
            /// @param inBeta
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> beta(const Shape& inShape, dtype inAlpha, dtype inBeta)
            {
                return detail::beta(generator_, inShape, inAlpha, inBeta);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the from the "binomial" distribution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.binomial.html#numpy.random.binomial
            ///
            /// @param inN (number of trials)
            /// @param inP (probablity of success [0, 1])
            /// @return NdArray
            ///
            template<typename dtype>
            dtype binomial(dtype inN, double inP = 0.5)
            {
                return detail::binomial(generator_, inN, inP);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from the "binomial" distribution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.binomial.html#numpy.random.binomial
            ///
            /// @param inShape
            /// @param inN (number of trials)
            /// @param inP (probablity of success [0, 1])
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> binomial(const Shape& inShape, dtype inN, double inP = 0.5)
            {
                return detail::binomial(generator_, inShape, inN, inP);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the from the "cauchy" distrubution.
            ///
            /// @param inMean: Mean value of the underlying normal distribution. Default is 0.
            /// @param inSigma: Standard deviation of the underlying normal distribution. Should be greater than zero.
            /// Default is 1.
            /// @return NdArray
            ///
            template<typename dtype>
            dtype cauchy(dtype inMean = 0, dtype inSigma = 1)
            {
                return detail::cauchy(generator_, inMean, inSigma);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from a "cauchy" distrubution.
            ///
            /// @param inShape
            /// @param inMean: Mean value of the underlying normal distribution. Default is 0.
            /// @param inSigma: Standard deviation of the underlying normal distribution. Should be greater than zero.
            /// Default is 1.
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> cauchy(const Shape& inShape, dtype inMean = 0, dtype inSigma = 1)
            {
                return detail::cauchy(generator_, inShape, inMean, inSigma);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the from the "chi square" distribution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.chisquare.html#numpy.random.chisquare
            ///
            /// @param inDof (independent random variables)
            /// @return NdArray
            ///
            template<typename dtype>
            dtype chiSquare(dtype inDof)
            {
                return detail::chiSquare(generator_, inDof);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from the "chi square" distribution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.chisquare.html#numpy.random.chisquare
            ///
            /// @param inShape
            /// @param inDof (independent random variables)
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> chiSquare(const Shape& inShape, dtype inDof)
            {
                return detail::chiSquare(generator_, inShape, inDof);
            }

            //============================================================================
            // Method Description:
            /// Chooses a random sample from an input array.
            ///
            /// @param inArray
            /// @return NdArray
            ///
            template<typename dtype>
            dtype choice(const NdArray<dtype>& inArray)
            {
                return detail::choice(generator_, inArray);
            }

            //============================================================================
            // Method Description:
            /// Chooses inNum random samples from an input array.
            ///
            /// @param inArray
            /// @param inNum
            /// @param replace: Whether the sample is with or without replacement
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> choice(const NdArray<dtype>& inArray, uint32 inNum, bool replace = true)
            {
                return detail::choice(generator_, inArray, inNum, replace);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the from the
            /// "discrete" distrubution.  It produces integers in the
            /// range [0, n) with the probability of producing each value
            /// is specified by the parameters of the distribution.
            ///
            /// @param inWeights
            /// @return NdArray
            ///
            template<typename dtype>
            dtype discrete(const NdArray<double>& inWeights)
            {
                return detail::discrete<dtype>(generator_, inWeights);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from a "discrete" distrubution.  It produces
            /// integers in the range [0, n) with the probability of
            /// producing each value is specified by the parameters
            /// of the distribution.
            ///
            /// @param inShape
            /// @param inWeights
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> discrete(const Shape& inShape, const NdArray<double>& inWeights)
            {
                return detail::discrete<dtype>(generator_, inShape, inWeights);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the "exponential" distrubution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.exponential.html#numpy.random.exponential
            ///
            /// @param inScaleValue (default 1)
            /// @return NdArray
            ///
            template<typename dtype>
            dtype exponential(dtype inScaleValue = 1)
            {
                return detail::exponential(generator_, inScaleValue);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from a "exponential" distrubution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.exponential.html#numpy.random.exponential
            ///
            /// @param inShape
            /// @param inScaleValue (default 1)
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> exponential(const Shape& inShape, dtype inScaleValue = 1)
            {
                return detail::exponential(generator_, inShape, inScaleValue);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the "extreme value" distrubution.
            ///
            /// @param inA (default 1)
            /// @param inB (default 1)
            /// @return NdArray
            ///
            template<typename dtype>
            dtype extremeValue(dtype inA = 1, dtype inB = 1)
            {
                return detail::extremeValue(generator_, inA, inB);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from a "extreme value" distrubution.
            ///
            /// @param inShape
            /// @param inA (default 1)
            /// @param inB (default 1)
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> extremeValue(const Shape& inShape, dtype inA = 1, dtype inB = 1)
            {
                return detail::extremeValue(generator_, inShape, inA, inB);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the "F" distrubution.
            ///
            /// NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.f.html#numpy.random.f
            ///
            /// @param inDofN: Degrees of freedom in numerator. Should be greater than zero.
            /// @param inDofD: Degrees of freedom in denominator. Should be greater than zero.
            /// @return NdArray
            ///
            template<typename dtype>
            dtype f(dtype inDofN, dtype inDofD)
            {
                return detail::f(generator_, inDofN, inDofD);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from a "F" distrubution.
            ///
            /// NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.f.html#numpy.random.f
            ///
            /// @param inShape
            /// @param inDofN: Degrees of freedom in numerator. Should be greater than zero.
            /// @param inDofD: Degrees of freedom in denominator. Should be greater than zero.
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> f(const Shape& inShape, dtype inDofN, dtype inDofD)
            {
                return detail::f(generator_, inShape, inDofN, inDofD);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the "gamma" distrubution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.gamma.html#numpy.random.gamma
            ///
            /// @param inGammaShape
            /// @param inScaleValue (default 1)
            /// @return NdArray
            ///
            template<typename dtype>
            dtype gamma(dtype inGammaShape, dtype inScaleValue = 1)
            {
                return detail::gamma(generator_, inGammaShape, inScaleValue);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from a "gamma" distrubution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.gamma.html#numpy.random.gamma
            ///
            /// @param inShape
            /// @param inGammaShape
            /// @param inScaleValue (default 1)
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> gamma(const Shape& inShape, dtype inGammaShape, dtype inScaleValue = 1)
            {
                return detail::gamma(generator_, inShape, inGammaShape, inScaleValue);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the "geometric" distrubution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.geometric.html#numpy.random.geometric
            ///
            /// @param inP (probablity of success [0, 1])
            /// @return NdArray
            ///
            template<typename dtype>
            dtype geometric(double inP = 0.5)
            {
                return detail::geometric<dtype>(generator_, inP);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from a "geometric" distrubution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.geometric.html#numpy.random.geometric
            ///
            /// @param inShape
            /// @param inP (probablity of success [0, 1])
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> geometric(const Shape& inShape, double inP = 0.5)
            {
                return detail::geometric<dtype>(generator_, inShape, inP);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the "laplace" distrubution.
            /// NOTE: Use of this function requires using the Boost includes.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.laplace.html#numpy.random.laplace
            ///
            /// @param inLoc: (The position, mu, of the distribution peak. Default is 0)
            /// @param inScale: (float optional the exponential decay. Default is 1)
            /// @return NdArray
            ///
            template<typename dtype>
            dtype laplace(dtype inLoc = 0, dtype inScale = 1)
            {
                return detail::laplace(generator_, inLoc, inScale);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from a "laplace" distrubution.
            /// NOTE: Use of this function requires using the Boost includes.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.laplace.html#numpy.random.laplace
            ///
            /// @param inShape
            /// @param inLoc: (The position, mu, of the distribution peak. Default is 0)
            /// @param inScale: (float optional the exponential decay. Default is 1)
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> laplace(const Shape& inShape, dtype inLoc = 0, dtype inScale = 1)
            {
                return detail::laplace(generator_, inShape, inLoc, inScale);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the "lognormal" distrubution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.lognormal.html#numpy.random.lognormal
            ///
            /// @param inMean: Mean value of the underlying normal distribution. Default is 0.
            /// @param inSigma: Standard deviation of the underlying normal distribution. Should be greater than zero.
            /// Default is 1.
            /// @return NdArray
            ///
            template<typename dtype>
            dtype lognormal(dtype inMean = 0, dtype inSigma = 1)
            {
                return detail::lognormal(generator_, inMean, inSigma);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from a "lognormal" distrubution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.lognormal.html#numpy.random.lognormal
            ///
            /// @param inShape
            /// @param inMean: Mean value of the underlying normal distribution. Default is 0.
            /// @param inSigma: Standard deviation of the underlying normal distribution. Should be greater than zero.
            /// Default is 1.
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> lognormal(const Shape& inShape, dtype inMean = 0, dtype inSigma = 1)
            {
                return detail::lognormal(generator_, inShape, inMean, inSigma);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the "negative Binomial" distribution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.negative_binomial.html#numpy.random.negative_binomial
            ///
            /// @param inN: number of trials
            /// @param inP: probablity of success [0, 1]
            /// @return NdArray
            ///
            template<typename dtype>
            dtype negativeBinomial(dtype inN, double inP = 0.5)
            {
                return detail::negativeBinomial(generator_, inN, inP);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from the "negative Binomial" distribution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.negative_binomial.html#numpy.random.negative_binomial
            ///
            /// @param inShape
            /// @param inN: number of trials
            /// @param inP: probablity of success [0, 1]
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> negativeBinomial(const Shape& inShape, dtype inN, double inP = 0.5)
            {
                return detail::negativeBinomial(generator_, inShape, inN, inP);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the "non central chi squared" distrubution.
            /// NOTE: Use of this function requires using the Boost includes.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.noncentral_chisquare.html#numpy.random.noncentral_chisquare
            ///
            /// @param inK (default 1)
            /// @param inLambda (default 1)
            /// @return NdArray
            ///
            template<typename dtype>
            dtype nonCentralChiSquared(dtype inK = 1, dtype inLambda = 1)
            {
                return detail::nonCentralChiSquared(generator_, inK, inLambda);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from a "non central chi squared" distrubution.
            /// NOTE: Use of this function requires using the Boost includes.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.noncentral_chisquare.html#numpy.random.noncentral_chisquare
            ///
            /// @param inShape
            /// @param inK (default 1)
            /// @param inLambda (default 1)
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> nonCentralChiSquared(const Shape& inShape, dtype inK = 1, dtype inLambda = 1)
            {
                return detail::nonCentralChiSquared(generator_, inShape, inK, inLambda);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the "normal" distrubution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html#numpy.random.normal
            ///
            /// @param inMean: Mean value of the underlying normal distribution. Default is 0.
            /// @param inSigma: Standard deviation of the underlying normal distribution. Should be greater than zero.
            /// Default is 1.
            /// @return NdArray
            ///
            template<typename dtype>
            dtype normal(dtype inMean = 0, dtype inSigma = 1)
            {
                return detail::normal(generator_, inMean, inSigma);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from a "normal" distrubution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html#numpy.random.normal
            ///
            /// @param inShape
            /// @param inMean: Mean value of the underlying normal distribution. Default is 0.
            /// @param inSigma: Standard deviation of the underlying normal distribution. Should be greater than zero.
            /// Default is 1.
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> normal(const Shape& inShape, dtype inMean = 0, dtype inSigma = 1)
            {
                return detail::normal(generator_, inShape, inMean, inSigma);
            }

            //============================================================================
            // Method Description:
            /// Randomly permute a sequence, or return a permuted range.
            /// If x is an integer, randomly permute np.arange(x).
            /// If x is an array, make a copy and shuffle the elements randomly.
            ///
            /// @param inValue
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> permutation(dtype inValue)
            {
                return detail::permutation(generator_, inValue);
            }

            //============================================================================
            // Method Description:
            /// Randomly permute a sequence, or return a permuted range.
            /// If x is an integer, randomly permute np.arange(x).
            /// If x is an array, make a copy and shuffle the elements randomly.
            ///
            /// @param inArray
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> permutation(const NdArray<dtype>& inArray)
            {
                return detail::permutation(generator_, inArray);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the uniform distribution over [0, 1).
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.rand.html#numpy.random.rand
            ///
            /// @return NdArray
            ///
            template<typename dtype>
            dtype rand()
            {
                return detail::rand<dtype>(generator_);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from a uniform distribution over [0, 1).
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.rand.html#numpy.random.rand
            ///
            /// @param inShape
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> rand(const Shape& inShape)
            {
                return detail::rand<dtype>(generator_, inShape);
            }

            //============================================================================
            // Method Description:
            /// Return a single random float from low (inclusive) to high (exclusive),
            /// with the given shape. If no high value is input then the range will
            /// go from [0, low).
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.ranf.html#numpy.random.ranf
            ///
            /// @param inLow
            /// @param inHigh default 0.
            /// @return NdArray
            ///
            template<typename dtype>
            dtype randFloat(dtype inLow, dtype inHigh = 0.0)
            {
                return detail::randFloat(generator_, inLow, inHigh);
            }

            //============================================================================
            // Method Description:
            /// Return random floats from low (inclusive) to high (exclusive),
            /// with the given shape. If no high value is input then the range will
            /// go from [0, low).
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.ranf.html#numpy.random.ranf
            ///
            /// @param inShape
            /// @param inLow
            /// @param inHigh default 0.
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> randFloat(const Shape& inShape, dtype inLow, dtype inHigh = 0.0)
            {
                return detail::randFloat(generator_, inShape, inLow, inHigh);
            }

            //============================================================================
            // Method Description:
            /// Return random integer from low (inclusive) to high (exclusive),
            /// with the given shape. If no high value is input then the range will
            /// go from [0, low).
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html#numpy.random.randint
            ///
            /// @param inLow
            /// @param inHigh default 0.
            /// @return NdArray
            ///
            template<typename dtype>
            dtype randInt(dtype inLow, dtype inHigh = 0)
            {
                return detail::randInt(generator_, inLow, inHigh);
            }

            //============================================================================
            // Method Description:
            /// Return random integers from low (inclusive) to high (exclusive),
            /// with the given shape. If no high value is input then the range will
            /// go from [0, low).
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html#numpy.random.randint
            ///
            /// @param inShape
            /// @param inLow
            /// @param inHigh default 0.
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> randInt(const Shape& inShape, dtype inLow, dtype inHigh = 0)
            {
                return detail::randInt(generator_, inShape, inLow, inHigh);
            }

            //============================================================================
            // Method Description:
            /// Returns a single random value sampled from the "standard normal" distribution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html#numpy.random.randn
            ///
            /// @return dtype
            ///
            template<typename dtype>
            dtype randN()
            {
                return detail::randN<dtype>(generator_);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from the "standard normal" distribution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html#numpy.random.randn
            ///
            /// @param inShape
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> randN(const Shape& inShape)
            {
                return detail::randN<dtype>(generator_, inShape);
            }

            //============================================================================
            // Method Description:
            /// Seed Constructor
            ///
            /// @param seed: the seed value
            ///
            void seed(int value) noexcept
            {
                generator_.seed(value);
            }

            //============================================================================
            // Method Description:
            /// Modify a sequence in-place by shuffling its contents.
            ///
            /// @param inArray
            ///
            template<typename dtype>
            void shuffle(NdArray<dtype>& inArray)
            {
                return detail::shuffle(generator_, inArray);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the "standard normal" distrubution with
            /// mean = 0 and std = 1
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.standard_normal.html#numpy.random.standard_normal
            ///
            /// @return NdArray
            ///
            template<typename dtype>
            dtype standardNormal()
            {
                return detail::standardNormal<dtype>(generator_);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from a "standard normal" distrubution with
            /// mean = 0 and std = 1
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.standard_normal.html#numpy.random.standard_normal
            ///
            /// @param inShape
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> standardNormal(const Shape& inShape)
            {
                return detail::standardNormal<dtype>(generator_, inShape);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the "student-T" distribution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.standard_t.html#numpy.random.standard_t
            ///
            /// @param inDof independent random variables
            /// @return NdArray
            ///
            template<typename dtype>
            dtype studentT(dtype inDof)
            {
                return detail::studentT(generator_, inDof);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from the "student-T" distribution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.standard_t.html#numpy.random.standard_t
            ///
            /// @param inShape
            /// @param inDof independent random variables
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> studentT(const Shape& inShape, dtype inDof)
            {
                return detail::studentT(generator_, inShape, inDof);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the "triangle" distribution.
            /// NOTE: Use of this function requires using the Boost includes.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.triangular.html#numpy.random.triangular
            ///
            /// @param inA
            /// @param inB
            /// @param inC
            /// @return NdArray
            ///
            template<typename dtype>
            dtype triangle(dtype inA = 0, dtype inB = 0.5, dtype inC = 1)
            {
                return detail::triangle(generator_, inA, inB, inC);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from the "triangle" distribution.
            /// NOTE: Use of this function requires using the Boost includes.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.triangular.html#numpy.random.triangular
            ///
            /// @param inShape
            /// @param inA
            /// @param inB
            /// @param inC
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> triangle(const Shape& inShape, dtype inA = 0, dtype inB = 0.5, dtype inC = 1)
            {
                return detail::triangle(generator_, inShape, inA, inB, inC);
            }

            //============================================================================
            // Method Description:
            /// Draw sample from a uniform distribution.
            ///
            /// Samples are uniformly distributed over the half -
            /// open interval[low, high) (includes low, but excludes high)
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.uniform.html#numpy.random.uniform
            ///
            /// @param inLow
            /// @param inHigh
            /// @return NdArray
            ///
            template<typename dtype>
            dtype uniform(dtype inLow, dtype inHigh)
            {
                return detail::uniform(generator_, inLow, inHigh);
            }

            //============================================================================
            // Method Description:
            /// Draw samples from a uniform distribution.
            ///
            /// Samples are uniformly distributed over the half -
            /// open interval[low, high) (includes low, but excludes high)
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.uniform.html#numpy.random.uniform
            ///
            /// @param inShape
            /// @param inLow
            /// @param inHigh
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> uniform(const Shape& inShape, dtype inLow, dtype inHigh)
            {
                return detail::uniform(generator_, inShape, inLow, inHigh);
            }

            //============================================================================
            // Method Description:
            /// Such a distribution produces random numbers uniformly
            /// distributed on the unit sphere of arbitrary dimension dim.
            /// NOTE: Use of this function requires using the Boost includes.
            ///
            /// @param inNumPoints
            /// @param inDims: dimension of the sphere (default 2)
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> uniformOnSphere(uint32 inNumPoints, uint32 inDims = 2)
            {
                return detail::uniformOnSphere<dtype>(generator_, inNumPoints, inDims);
            }

            //============================================================================
            // Method Description:
            /// Single random value sampled from the  "weibull" distribution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.weibull.html#numpy.random.weibull
            ///
            /// @param inA (default 1)
            /// @param inB (default 1)
            /// @return NdArray
            ///
            template<typename dtype>
            dtype weibull(dtype inA = 1, dtype inB = 1)
            {
                return detail::weibull(generator_, inA, inB);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from the "weibull" distribution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.weibull.html#numpy.random.weibull
            ///
            /// @param inShape
            /// @param inA (default 1)
            /// @param inB (default 1)
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> weibull(const Shape& inShape, dtype inA = 1, dtype inB = 1)
            {
                return detail::weibull(generator_, inShape, inA, inB);
            }

        private:
            GeneratorType generator_{};
        };
    } // namespace random
} // namespace nc
