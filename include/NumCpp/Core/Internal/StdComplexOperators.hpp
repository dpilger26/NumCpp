/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.4
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
/// Additional operator for std::complex<T>
///
#pragma once

#include <complex>

namespace nc
{
    //============================================================================
    // Method Description:
    ///	Less than operator for std::complex<T>
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     bool true if lhs < rhs
    ///
    template<typename T>
    bool operator<(const std::complex<T>& lhs, const std::complex<T>& rhs)
    {
        if (lhs.real() != rhs.real())
        {
            return lhs.real() < rhs.real();
        }
        else
        {
            return lhs.imag() < rhs.imag();
        }
    }

    //============================================================================
    // Method Description:
    ///	Less than or equal operator for std::complex<T>
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     bool true if lhs <= rhs
    ///
    template<typename T>
    bool operator<=(const std::complex<T>& lhs, const std::complex<T>& rhs)
    {
        return lhs < rhs || lhs == rhs;
    }

    //============================================================================
    // Method Description:
    ///	Greater than operator for std::complex<T>
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     bool true if lhs > rhs
    ///
    template<typename T>
    bool operator>(const std::complex<T>& lhs, const std::complex<T>& rhs)
    {
        return !(lhs < rhs);
    }

    //============================================================================
    // Method Description:
    ///	Greater than or equal operator for std::complex<T>
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     bool true if lhs >= rhs
    ///
    template<typename T>
    bool operator>=(const std::complex<T>& lhs, const std::complex<T>& rhs)
    {
        return lhs > rhs || lhs == rhs;
    }
}
