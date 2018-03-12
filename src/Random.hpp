// Copyright 2018 David Pilger
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files(the "Software"), to deal in the Software 
// without restriction, including without limitation the rights to use, copy, modify, 
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
// permit persons to whom the Software is furnished to do so, subject to the following 
// conditions :
//
// The above copyright notice and this permission notice shall be included in all copies 
// or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
// PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
// FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
// DEALINGS IN THE SOFTWARE.

#pragma once

#include"Types.hpp"
#include"Shape.hpp"
#include"NdArray.hpp"

#include"boost/random.hpp"

namespace NumC
{
	//================================Random Namespace=============================
	namespace Random
	{
		// global generator function
		boost::random::mt19937 generator;

		//============================================================================
		// Method Description: 
		//						Create an array of the given shape and populate it with 
		//						random samples from the “beta” distribution.
		//		
		// Inputs:
		//				alpha
		//				beta
		//				Shape
		// Outputs:
		//				None
		//
		template<typename dtype>
		NdArray<dtype> beta(const Shape& inShape, dtype inAlpha, dtype inBeta)
		{
			NdArray<dtype> returnArray(inShape);

			boost::random::beta_distribution<dtype> dist(inAlpha, inBeta);
			for (uint32 i = 0; i < returnArray.size(); ++i)
			{
				returnArray[i] = dist(generator);
			}

			return std::move(returnArray);

		}

		//============================================================================
		// Method Description: 
		//						Create an array of the given shape and populate it with 
		//						random samples from the “binomial” distribution.
		//		
		// Inputs:
		//				number of trials
		//				probablity of success [0, 1]
		//				Shape
		// Outputs:
		//				None
		//
		template<typename dtype>
		NdArray<dtype> binomial(const Shape& inShape, uint32 inN, double inP)
		{
			// only works with integer input types
			static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: binomial: can only use with integer types.");

			if (inP < 0 || inP > 1)
			{
				throw std::invalid_argument("Error: binomial: input probability of sucess must be of the range [0, 1].");
			}

			NdArray<dtype> returnArray(inShape);

			boost::random::binomial_distribution<dtype, double> dist(inN, inP);
			for (uint32 i = 0; i < returnArray.size(); ++i)
			{
				returnArray[i] = dist(generator);
			}

			return std::move(returnArray);
		}

		//============================================================================
		// Method Description: 
		//						Create an array of the given shape and populate it with 
		//						random samples from the “chi square” distribution.
		//		
		// Inputs:
		//				df independent random variables
		//				Shape
		// Outputs:
		//				None
		//
		template<typename dtype>
		NdArray<dtype> chisquare(const Shape& inShape, uint32 inDf)
		{
			NdArray<dtype> returnArray(inShape);

			boost::random::chi_squared_distribution<dtype> dist(inDf);
			for (uint32 i = 0; i < returnArray.size(); ++i)
			{
				returnArray[i] = dist(generator);
			}

			return std::move(returnArray);
		}

		//============================================================================
		// Method Description: 
		//						Generates a random sample from an input array
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				None
		//
		template<typename dtype>
		dtype choice(const NdArray<dtype>& inArray)
		{
			uint32 randIdx = randint<uint32>(Shape(1), 0, inArray.size()).item();
			return inArray[randIdx];
		}

		//============================================================================
		// Method Description: 
		//						Create an array of the given shape and populate it with 
		//						random samples from a "exponential" distrubution.
		//		
		// Inputs:
		//				scale value, default 1
		//				Shape
		// Outputs:
		//				None
		//
		template<typename dtype>
		NdArray<dtype> exponential(const Shape& inShape, dtype inScaleValue=1)
		{
			NdArray<dtype> returnArray(inShape);

			boost::random::exponential_distribution<dtype> dist(inScaleValue);
			for (uint32 i = 0; i < returnArray.size(); ++i)
			{
				returnArray[i] = dist(generator);
			}

			return std::move(returnArray);
		}

		//============================================================================
		// Method Description: 
		//						Create an array of the given shape and populate it with 
		//						random samples from a "F" distrubution.
		//		
		// Inputs:
		//				Degrees of freedom in numerator. Should be greater than zero.
		//				Degrees of freedom in denominator. Should be greater than zero.
		//				Shape
		// Outputs:
		//				None
		//
		template<typename dtype>
		NdArray<dtype> f(const Shape& inShape, dtype inDofN, dtype inDofD)
		{
			if (inDofN < 0)
			{
				throw std::invalid_argument("Error: f: numerator degrees of freedom should be greater than zero.");
			}

			if (inDofD < 0)
			{
				throw std::invalid_argument("Error: f: denominator degrees of freedom should be greater than zero.");
			}

			NdArray<dtype> returnArray(inShape);

			boost::random::fisher_f_distribution<dtype> dist(inDofN, inDofD);
			for (uint32 i = 0; i < returnArray.size(); ++i)
			{
				returnArray[i] = dist(generator);
			}

			return std::move(returnArray);
		}

		//============================================================================
		// Method Description: 
		//						Create an array of the given shape and populate it with 
		//						random samples from a "F" distrubution.
		//		
		// Inputs:
		//				Scale, default 1
		//				Gamma shape
		//				Shape
		// Outputs:
		//				None
		//
		template<typename dtype>
		NdArray<dtype> gamma(const Shape& inShape, dtype inGammaShape, dtype inScaleValue=1)
		{
			if (inGammaShape < 0)
			{
				throw std::invalid_argument("Error: gamma: input gamma shape should be greater than zero.");
			}

			if (inScaleValue < 0)
			{
				throw std::invalid_argument("Error: gamma: input scale should be greater than zero.");
			}

			NdArray<dtype> returnArray(inShape);

			boost::random::gamma_distribution<dtype> dist(inGammaShape, inScaleValue);
			for (uint32 i = 0; i < returnArray.size(); ++i)
			{
				returnArray[i] = dist(generator);
			}

			return std::move(returnArray);
		}

		//============================================================================
		// Method Description: 
		//						Create an array of the given shape and populate it with 
		//						random samples from a uniform distribution over [0, 1).
		//		
		// Inputs:
		//				Shape
		// Outputs:
		//				None
		//
		template<typename dtype>
		NdArray<dtype> rand(const Shape& inShape)
		{
			NdArray<dtype> returnArray(inShape);

			boost::random::uniform_01<dtype> dist;
			for (uint32 i = 0; i < returnArray.size(); ++i)
			{
				returnArray[i] = dist(generator);
			}

			return std::move(returnArray);
		}

		//============================================================================
		// Method Description: 
		//						Return random integers from low (inclusive) to high (exclusive), 
		//						with the given shape
		//		
		// Inputs:
		//				low value
		//				high value
		//				Shape
		// Outputs:
		//				None
		//
		template<typename dtype>
		NdArray<dtype> randint(const Shape& inShape, dtype inLow, dtype inHigh)
		{
			// only works with integer input types
			static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: randint: can only use with integer types.");

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
				returnArray[i] = dist(generator);
			}

			return std::move(returnArray);
		}

		//============================================================================
		// Method Description: 
		//						Create an array of the given shape and populate it with 
		//						random samples from the “standard normal” distribution.
		//		
		// Inputs:
		//				Shape
		// Outputs:
		//				None
		//
		template<typename dtype>
		NdArray<dtype> randn(const Shape& inShape)
		{
			NdArray<dtype> returnArray(inShape);

			boost::random::normal_distribution<dtype> dist;
			for (uint32 i = 0; i < returnArray.size(); ++i)
			{
				returnArray[i] = dist(generator);
			}

			return std::move(returnArray);
		}

		//============================================================================
		// Method Description: 
		//						Seeds the random number generator
		//		
		// Inputs:
		//				seed
		// Outputs:
		//				None
		//
		void seed(uint32 inSeed)
		{
			generator.seed(inSeed);
		}
	}
}