/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.0
///
/// @section LICENSE
/// Copyright 2018 David Pilger
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
/// @section DESCRIPTION
/// A module for generating random numbers
///
#pragma once

#include"NumCpp/Methods.hpp"
#include"NumCpp/NdArray.hpp"
#include"NumCpp/Shape.hpp"
#include"NumCpp/Types.hpp"

#include"boost/random.hpp"

#include<algorithm>
#include<vector>

namespace NC
{
    /// generator function
    boost::random::mt19937 generator_;

    //================================Random Class=============================
    /// A class for generating random numbers
    template<typename dtype = int32>
    class Random
    {
    public:
        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from the “bernoulli” distribution.
        ///		
        /// @param				Shape
        /// @param				probablity of success [0, 1]
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> bernoulli(const Shape& inShape, dtype inP)
        {
            if (inP < 0 || inP > 1)
            {
                throw std::invalid_argument("Error: bernoulli: input probability of sucess must be of the range [0, 1].");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::bernoulli_distribution<dtype> dist(inP);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from the “beta” distribution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.beta.html#numpy.random.beta
        ///		
        /// @param				Shape
        /// @param				alpha
        /// @param				beta
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> beta(const Shape& inShape, dtype inAlpha, dtype inBeta)
        {
            if (inAlpha < 0)
            {
                throw std::invalid_argument("Error: beta: input alpha must be greater than zero.");
            }

            if (inBeta < 0)
            {
                throw std::invalid_argument("Error: beta: input beta must be greater than zero.");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::beta_distribution<dtype> dist(inAlpha, inBeta);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from the “binomial” distribution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.binomial.html#numpy.random.binomial
        ///		
        /// @param				Shape
        /// @param				number of trials
        /// @param				probablity of success [0, 1]
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> binomial(const Shape& inShape, dtype inN, double inP = 0.5)
        {
            // only works with integer input types
            static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: binomial: can only use with integer types.");

            if (inN < 0)
            {
                throw std::invalid_argument("Error: binomial: input number of trials must be greater than or equal to zero.");
            }

            if (inP < 0 || inP > 1)
            {
                throw std::invalid_argument("Error: binomial: input probability of sucess must be of the range [0, 1].");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::binomial_distribution<dtype, double> dist(inN, inP);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from the “chi square” distribution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.chisquare.html#numpy.random.chisquare
        ///		
        /// @param				Shape
        /// @param				df independent random variables
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> chiSquare(const Shape& inShape, dtype inDof)
        {
            if (inDof <= 0)
            {
                throw std::invalid_argument("Error: chisquare: numerator degrees of freedom must be greater than zero.");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::chi_squared_distribution<dtype> dist(inDof);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Generates a random sample from an input array
        ///		
        /// @param
        ///				NdArray
        /// @return
        ///				NdArray
        ///
        static dtype choice(const NdArray<dtype>& inArray)
        {
            uint32 randIdx = Random<uint32>::randInt(Shape(1), 0, inArray.size()).item();
            return inArray[randIdx];
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from a "cauchy" distrubution.
        ///		
        /// @param				mean: Mean value of the underlying normal distribution. Default is 0.
        /// @param				sigma, Standard deviation of the underlying normal distribution. Should be greater than zero. Default is 1.
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> cauchy(const Shape& inShape, dtype inMean = 0, dtype inSigma = 1)
        {
            if (inSigma <= 0)
            {
                throw std::invalid_argument("Error: cauchy: input sigma must be greater than zero.");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::cauchy_distribution<dtype> dist(inMean, inSigma);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from a "discrete" distrubution.  It produces
        ///						integers in the range [0, n) with the probability of 
        ///						producing each value is specified by the parameters 
        ///						of the distribution.
        ///		
        /// @param
        ///				NdArray of weights, 
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> discrete(const Shape& inShape, const NdArray<double>& inWeights)
        {
            // only works with integer input types
            static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: discrete: can only use with integer types.");

            NdArray<dtype> returnArray(inShape);

            boost::random::discrete_distribution<dtype> dist(inWeights.cbegin(), inWeights.cend());
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from a "exponential" distrubution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.exponential.html#numpy.random.exponential
        ///		
        /// @param				Shape
        /// @param				scale value, default 1
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> exponential(const Shape& inShape, dtype inScaleValue = 1)
        {
            NdArray<dtype> returnArray(inShape);

            boost::random::exponential_distribution<dtype> dist(inScaleValue);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from a "extreme value" distrubution.
        ///		
        /// @param				Shape
        /// @param				a, default 1
        /// @param				b, default 1
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> extremeValue(const Shape& inShape, dtype inA = 1, dtype inB = 1)
        {
            if (inA <= 0)
            {
                throw std::invalid_argument("Error: extremeValue: input a must be greater than zero.");
            }

            if (inB <= 0)
            {
                throw std::invalid_argument("Error: extremeValue: input b must be greater than zero.");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::extreme_value_distribution<dtype> dist(inA, inB);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from a "F" distrubution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.f.html#numpy.random.f
        ///		
        /// @param				Shape
        /// @param				Degrees of freedom in numerator. Should be greater than zero.
        /// @param				Degrees of freedom in denominator. Should be greater than zero.
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> f(const Shape& inShape, dtype inDofN, dtype inDofD)
        {
            if (inDofN <= 0)
            {
                throw std::invalid_argument("Error: f: numerator degrees of freedom should be greater than zero.");
            }

            if (inDofD <= 0)
            {
                throw std::invalid_argument("Error: f: denominator degrees of freedom should be greater than zero.");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::fisher_f_distribution<dtype> dist(inDofN, inDofD);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from a "gamma" distrubution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.gamma.html#numpy.random.gamma
        ///		
        /// @param				Shape
        /// @param			    Scale, default 1
        /// @param				Gamma shape
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> gamma(const Shape& inShape, dtype inGammaShape, dtype inScaleValue = 1)
        {
            if (inGammaShape <= 0)
            {
                throw std::invalid_argument("Error: gamma: input gamma shape should be greater than zero.");
            }

            if (inScaleValue <= 0)
            {
                throw std::invalid_argument("Error: gamma: input scale should be greater than zero.");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::gamma_distribution<dtype> dist(inGammaShape, inScaleValue);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from a "geometric" distrubution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.geometric.html#numpy.random.geometric
        ///		
        /// @param				Shape
        /// @param				probablity of success [0, 1]
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> geometric(const Shape& inShape, double inP = 0.5)
        {
            // only works with integer input types
            static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: geometric: can only use with integer types.");

            if (inP < 0 || inP > 1)
            {
                throw std::invalid_argument("Error: geometric: input probability of sucess must be of the range [0, 1].");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::geometric_distribution<dtype, double> dist(inP);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from a "laplace" distrubution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.laplace.html#numpy.random.laplace
        ///		
        /// @param				inLoc: The position, mu, of the distribution peak. Default is 0.
        /// @param				inScale: float  optional, the exponential decay. Default is 1.
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> laplace(const Shape& inShape, dtype inLoc = 0, dtype inScale = 1)
        {
            NdArray<dtype> returnArray(inShape);

            boost::random::laplace_distribution<dtype> dist(inLoc, inScale);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from a "lognormal" distrubution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.lognormal.html#numpy.random.lognormal
        ///		
        /// @param				mean: Mean value of the underlying normal distribution. Default is 0.
        /// @param				sigma, Standard deviation of the underlying normal distribution. Should be greater than zero. Default is 1.
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> lognormal(const Shape& inShape, dtype inMean = 0, dtype inSigma = 1)
        {
            if (inSigma <= 0)
            {
                throw std::invalid_argument("Error: lognormal: input sigma must be greater than zero.");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::lognormal_distribution<dtype> dist(inMean, inSigma);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from the “negative Binomial” distribution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.negative_binomial.html#numpy.random.negative_binomial
        ///		
        /// @param				Shape
        /// @param				number of trials
        /// @param				probablity of success [0, 1]
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> negativeBinomial(const Shape& inShape, dtype inN, double inP = 0.5)
        {
            // only works with integer input types
            static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: negativeBinomial: can only use with integer types.");

            if (inN < 0)
            {
                throw std::invalid_argument("Error: negativeBinomial: input number of trials must be greater than or equal to zero.");
            }

            if (inP < 0 || inP > 1)
            {
                throw std::invalid_argument("Error: negativeBinomial: input probability of sucess must be of the range [0, 1].");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::negative_binomial_distribution<dtype, double> dist(inN, inP);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from a "non central chi squared" distrubution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.noncentral_chisquare.html#numpy.random.noncentral_chisquare
        ///		
        /// @param				Shape
        /// @param				k, default 1
        /// @param				lambda, default 1
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> nonCentralChiSquared(const Shape& inShape, dtype inK = 1, dtype inLambda = 1)
        {
            if (inK <= 0)
            {
                throw std::invalid_argument("Error: nonCentralChiSquared: input k must be greater than zero.");
            }

            if (inLambda <= 0)
            {
                throw std::invalid_argument("Error: nonCentralChiSquared: input lambda must be greater than zero.");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::non_central_chi_squared_distribution<dtype> dist(inK, inLambda);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from a "normal" distrubution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html#numpy.random.normal
        ///		
        /// @param				mean: Mean value of the underlying normal distribution. Default is 0.
        /// @param  			sigma, Standard deviation of the underlying normal distribution. Should be greater than zero. Default is 1.
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> normal(const Shape& inShape, dtype inMean = 0, dtype inSigma = 1)
        {
            if (inSigma <= 0)
            {
                throw std::invalid_argument("Error: cauchy: input sigma must be greater than zero.");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::normal_distribution<dtype> dist(inMean, inSigma);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Randomly permute a sequence, or return a permuted range.
        ///						If x is an integer, randomly permute np.arange(x). 
        ///						If x is an array, make a copy and shuffle the elements randomly.
        ///		
        /// @param
        ///				value
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> permutation(dtype inValue)
        {
            NdArray<dtype> returnArray = Methods<dtype>::arange(inValue);
            std::random_shuffle(returnArray.begin(), returnArray.end());
            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Randomly permute a sequence, or return a permuted range.
        ///						If x is an integer, randomly permute np.arange(x). 
        ///						If x is an array, make a copy and shuffle the elements randomly.
        ///		
        /// @param
        ///				NdArray
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> permutation(const NdArray<dtype>& inArray)
        {
            NdArray<dtype> returnArray(inArray);
            std::random_shuffle(returnArray.begin(), returnArray.end());
            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from the “poisson” distribution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.poisson.html#numpy.random.poisson
        ///		
        /// @param				Shape
        /// @param				mean, default 1
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> poisson(const Shape& inShape, double inMean = 1)
        {
            if (inMean <= 0)
            {
                throw std::invalid_argument("Error: poisson: input mean must be greater than zero.");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::poisson_distribution<dtype, double> dist(inMean);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from a uniform distribution over [0, 1).
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.rand.html#numpy.random.rand
        ///		
        /// @param
        ///				Shape
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> rand(const Shape& inShape)
        {
            NdArray<dtype> returnArray(inShape);

            boost::random::uniform_01<dtype> dist;
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Return random floats from low (inclusive) to high (exclusive), 
        ///						with the given shape
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.ranf.html#numpy.random.ranf
        ///		
        /// @param				Shape
        /// @param  			low value
        /// @param				high value
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> randFloat(const Shape& inShape, dtype inLow, dtype inHigh)
        {
            if (inLow == inHigh)
            {
                throw std::invalid_argument("Error: randFloat: input low value must be less than the input high value.");
            }
            else if (inLow > inHigh - DtypeInfo<dtype>::epsilon())
            {
                std::swap(inLow, inHigh);
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::uniform_real_distribution<dtype> dist(inLow, inHigh - DtypeInfo<dtype>::epsilon());
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Return random integers from low (inclusive) to high (exclusive), 
        ///						with the given shape
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html#numpy.random.randint
        ///		
        /// @param				Shape
        /// @param				low value
        /// @param				high value
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> randInt(const Shape& inShape, dtype inLow, dtype inHigh)
        {
            // only works with integer input types
            static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: randInt: can only use with integer types.");

            if (inLow == inHigh)
            {
                throw std::invalid_argument("Error: randint: input low value must be less than the input high value.");
            }
            else if (inLow > inHigh - 1)
            {
                std::swap(inLow, inHigh);
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::uniform_int_distribution<dtype> dist(inLow, inHigh - 1);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from the “standard normal” distribution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html#numpy.random.randn
        ///		
        /// @param
        ///				Shape
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> randN(const Shape& inShape)
        {
            NdArray<dtype> returnArray(inShape);

            boost::random::normal_distribution<dtype> dist;
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Seeds the random number generator_
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html#numpy.random.seed
        ///		
        /// @param
        ///				seed
        /// @return
        ///				None
        ///
        static void seed(uint32 inSeed)
        {
            generator_.seed(inSeed);
        }

        //============================================================================
        // Method Description: 
        ///						Modify a sequence in-place by shuffling its contents.
        ///		
        /// @param
        ///				NdArray
        /// @return
        ///				None
        ///
        static void shuffle(NdArray<dtype>& inArray)
        {
            std::random_shuffle(inArray.begin(), inArray.end());
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from a "standard normal" distrubution with 
        ///						mean = 0 and std = 1
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.standard_normal.html#numpy.random.standard_normal
        ///		
        /// @param
        ///				Shape
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> standardNormal(const Shape& inShape)
        {
            return std::move(normal(inShape, 0, 1));
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from the “student-T” distribution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.standard_t.html#numpy.random.standard_t
        ///		
        /// @param				Shape
        /// @param				df independent random variables
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> studentT(const Shape& inShape, dtype inDof)
        {
            if (inDof <= 0)
            {
                throw std::invalid_argument("Error: studentT: degrees of freedom must be greater than zero.");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::student_t_distribution<dtype> dist(inDof);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from the “triangle” distribution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.triangular.html#numpy.random.triangular
        ///		
        /// @param				Shape
        /// @param				a
        /// @param				b
        /// @param				c
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> triangle(const Shape& inShape, dtype inA = 0, dtype inB = 0.5, dtype inC = 1)
        {
            if (inA < 0)
            {
                throw std::invalid_argument("Error: triangle: input A must be greater than or equal to zero.");
            }

            if (inB < 0)
            {
                throw std::invalid_argument("Error: triangle: input B must be greater than or equal to zero.");
            }

            if (inC < 0)
            {
                throw std::invalid_argument("Error: triangle: input C must be greater than or equal to zero.");
            }

            bool aLessB = inA <= inB;
            bool bLessC = inB <= inC;
            if (!(aLessB && bLessC))
            {
                throw std::invalid_argument("Error: triangle: inputs must be a <= b <= c.");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::triangle_distribution<dtype> dist(inA, inB, inC);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Draw samples from a uniform distribution.
        ///
        ///						Samples are uniformly distributed over the half - 
        ///						open interval[low, high) (includes low, but excludes high)
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.uniform.html#numpy.random.uniform
        ///		
        /// @param				Shape
        /// @param				low value
        /// @param				high value
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> uniform(const Shape& inShape, dtype inLow, dtype inHigh)
        {
            return std::move(randFloat(inShape, inLow, inHigh));
        }

        //============================================================================
        // Method Description: 
        ///						Such a distribution produces random numbers uniformly
        ///						distributed on the unit sphere of arbitrary dimension dim.
        ///		
        /// @param				number of points
        /// @param				dimension of the sphere, default 2
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> uniformOnSphere(uint32 inNumPoints, uint32 inDims = 2)
        {
            if (inDims < 0)
            {
                throw std::invalid_argument("Error: uniformOnSphere: input dimension must be greater than or equal to zero.");
            }

            boost::random::uniform_on_sphere<dtype> dist(inDims);

            NdArray<dtype> returnArray(inNumPoints, inDims);
            for (uint32 i = 0; i < inNumPoints; ++i)
            {
                std::vector<dtype> point = dist(generator_);
                for (uint32 dim = 0; dim < inDims; ++dim)
                {
                    returnArray(i, dim) = point[dim];
                }
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Create an array of the given shape and populate it with 
        ///						random samples from the "weibull" distribution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.weibull.html#numpy.random.weibull
        ///		
        /// @param				Shape
        /// @param				a, default 1
        /// @param				b, default 1
        /// @return
        ///				NdArray
        ///
        static NdArray<dtype> weibull(const Shape& inShape, dtype inA = 1, dtype inB = 1)
        {
            if (inA <= 0)
            {
                throw std::invalid_argument("Error: weibull: input a must be greater than zero.");
            }

            if (inB <= 0)
            {
                throw std::invalid_argument("Error: weibull: input b must be greater than zero.");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::weibull_distribution<dtype> dist(inA, inB);
            for (uint32 i = 0; i < returnArray.size(); ++i)
            {
                returnArray[i] = dist(generator_);
            }

            return std::move(returnArray);
        }
    };
}
