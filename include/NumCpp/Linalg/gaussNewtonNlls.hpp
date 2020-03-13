/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.3
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
/// The Gauss–Newton algorithm is used to solve non-linear least squares problems.
/// It is a modification of Newton's method for finding a minimum of a function.
/// 
/// https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
#pragma once

#include "NumCpp/Core/Error.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Core/TypeTraits.hpp"
#include "NumCpp/NdArray.hpp"

#include <array>
#include <string>
#include <type_traits>

namespace nc
{
    namespace linalg
    {
        //============================================================================
        // Method Description:
        /// The Gauss–Newton algorithm is used to solve non-linear least squares problems.
        /// It is a modification of Newton's method for finding a minimum of a function.
        /// https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
        ///
        /// @param      values: the values to be fitted
        /// @param      initialGuess: the initial guess of the parameters to be solved for
        /// @param      function: the function that is being fit
        /// @param      derivatives: vector of std::functions to calculate the function 
        ///                          derivatives   
        /// @param      numIterations: the number of iterations to perform, default 4.
        ///
        /// @return     NdArray of solved parameter values
        ///
        template<typename dtype, typename ...Args,
            std::enable_if_t<std::is_arithmetic<dtype>::value, int> = 0,
            std::enable_if_t<all_arithmetic<Args...>::value, int> = 0,
            std::enable_if_t<all_same<dtype, Args...>::value, int> = 0
        >
        NdArray<double> gaussNewtonNlls(
            const NdArray<dtype> values,
            const std::array<dtype, sizeof...(Args)>& initialGuess,
            const std::function<dtype(const NdArray<dtype>&, Args...)>& function,
            const std::array<std::function<dtype(const NdArray<dtype>&, Args...)>, sizeof...(Args)>& derivatives,
            const uint32 numIterations = 4)
        {
            // need to get Args... values into an NdArray somehow???

            for (uint32 i = 1; i <= numIterations; ++i)
            {
                // build the jacobian and residials matrices and perform gauss-newton
            }

            return {};
        }
    }
}
