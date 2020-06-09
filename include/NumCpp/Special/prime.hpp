/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.0.0
///
/// @section License
/// Copyright 2020 David Pilger
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
/// @section Description
/// Special Functions
///
#pragma once

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"

#include "boost/math/special_functions/prime.hpp"

#include <string>

namespace nc
{
    namespace special
    {
        //============================================================================
        // Method Description:
        /// The function prime provides fast table lookup to the first 10000 prime numbers
        /// (starting from 2 as the zeroth prime: as 1 isn't terribly useful in practice)
        ///
        /// @param
        ///				n: the nth prime number to return
        /// @return
        ///				uint32
        ///
        inline uint32 prime(uint32 n)
        {
            if (n > boost::math::max_prime)
            {
                THROW_INVALID_ARGUMENT_ERROR("input n must be less than or equal to " + std::to_string(boost::math::max_prime));
            }

            return boost::math::prime(n);
        }

        //============================================================================
        // Method Description:
        /// The function prime provides fast table lookup to the first 10000 prime numbers
        /// (starting from 2 as the zeroth prime: as 1 isn't terribly useful in practice)
        ///
        /// @param
        ///				inArray
        /// @return
        ///				NdArray<uint32, Alloc>
        ///
        template<class Alloc>
        NdArray<uint32, Alloc> prime(const NdArray<uint32, Alloc>& inArray)
        {
            NdArray<uint32, Alloc> returnArray(inArray.shape());

            stl_algorithms::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
                [](uint32 inValue) -> uint32
                { 
                    return prime(inValue); 
                });

            return returnArray;
        }
    }
}
