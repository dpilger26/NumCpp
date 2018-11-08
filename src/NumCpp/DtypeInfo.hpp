/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.0
///
/// @section License
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
/// @section Description
/// Holds info about the dtype
///
#pragma once

#include<limits>

namespace nc
{
    //================================================================================
    ///						Holds info about the dtype
    template<typename dtype>
    class DtypeInfo
    {
    public:
        //============================================================================
        ///						For integer types: number of non-sign bits in the representation.
        ///						For floating types : number of digits(in radix base) in the mantissa
        ///
        /// @return     number of bits
        ///
        static constexpr dtype bits()
        {
            return std::numeric_limits<dtype>::digits;
        }

        //============================================================================
        ///						Machine epsilon (the difference between 1 and the least
        ///						value greater than 1 that is representable).
        ///
        /// @return     dtype
        ///
        static constexpr dtype epsilon()
        {
            return std::numeric_limits<dtype>::epsilon();
        }

        //============================================================================
        ///						True if type is integer.
        ///
        /// @return     bool
        ///
        static constexpr bool isInteger()
        {
            return std::numeric_limits<dtype>::is_integer;
        }

        //============================================================================
        ///						True if type is signed.
        ///
        /// @return     bool
        ///
        static constexpr bool isSigned()
        {
            return std::numeric_limits<dtype>::is_signed;
        }

        //============================================================================
        ///						Returns the minimum value of the dtype
        ///
        /// @return     min value
        ///
        static constexpr dtype min()
        {
            return std::numeric_limits<dtype>::min();
        }

        //============================================================================
        ///						Returns the maximum value of the dtype
        ///
        /// @return     max value
        ///
        static constexpr dtype max()
        {
            return std::numeric_limits<dtype>::max();
        }
    };
}
