/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2023 David Pilger
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
/// Special Functions
///
#pragma once

#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Special/factorial.hpp"

#ifndef NUMCPP_NO_USE_BOOST
#include "boost/math/special_functions/factorials.hpp"
#endif

namespace nc::special
{
    //============================================================================
    // Method Description:
    /// Returns the number of permutaions of n choose r. P(n, r)
    ///
    /// @param n: the total number of items
    /// @param r: the number of items taken
    /// @return double
    ///
    inline double pnr(uint32 n, uint32 r)
    {
        if (r > n)
        {
            return 0.;
        }
        else if (r == n)
        {
            return factorial(n);
        }

        double combinations = 1.;

#ifndef NUMCPP_NO_USE_BOOST
        if (n <= boost::math::max_factorial<double>::value)
        {
            const double nFactorial      = factorial(n);
            const double nMinusRFactoral = factorial(n - r);

            combinations = nFactorial / nMinusRFactoral;
        }
        else
        {
#endif
            const uint32 lower = n - r + 1;
            combinations       = static_cast<double>(lower);
            for (uint32 i = lower + 1; i <= n; ++i)
            {
                combinations *= static_cast<double>(i);
            }
#ifndef NUMCPP_NO_USE_BOOST
        }
#endif

        return combinations;
    }
} // namespace nc::special
