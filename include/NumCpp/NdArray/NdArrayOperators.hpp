/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.2.0
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
/// Operators for the NdArray class
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StdComplexOperators.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Functions/complex.hpp"
#include "NumCpp/NdArray/NdArrayCore.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"

#include <algorithm>
#include <complex>
#include <functional>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Adds the elements of two arrays (1)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator+=(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        stl_algorithms::transform(lhs.begin(), lhs.end(), 
            rhs.cbegin(), lhs.begin(), std::plus<dtype>());

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Adds the elements of two arrays (2)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>>& operator+=(NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        const auto function = [](std::complex<dtype>& val1, dtype val2) -> std::complex<dtype>
        {
            return val1 + val2;
        };

        stl_algorithms::transform(lhs.begin(), lhs.end(), 
            rhs.cbegin(), lhs.begin(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Adds the scaler to the array (3)
    ///
    /// @param      lhs
    ///	@param      rhs
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator+=(NdArray<dtype>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [rhs](dtype& value)  -> dtype
        {
            return value += rhs;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Adds the scaler to the array (4)
    ///
    /// @param      lhs
    ///	@param      rhs
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>>& operator+=(NdArray<std::complex<dtype>>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](std::complex<dtype>& value)  -> std::complex<dtype>
        {
            return value += rhs;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Adds the elements of two arrays (1)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator+(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        NdArray<dtype> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), 
            rhs.cbegin(), returnArray.begin(), std::plus<dtype>());

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Adds the elements of two arrays (2)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator+(const NdArray<dtype>& lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        const auto function = [](dtype val1, const std::complex<dtype>& val2) -> std::complex<dtype>
        {
            return val1 + val2;
        };

        NdArray<std::complex<dtype>> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), 
            rhs.cbegin(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Adds the elements of two arrays (3)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator+(const NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return rhs + lhs;
    }

    //============================================================================
    // Method Description:
    ///						Adds the scaler to the array (4)
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator+(const NdArray<dtype>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [rhs](dtype value)  -> dtype 
        {
            return value + rhs;
        };

        NdArray<dtype> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Adds the scaler to the array (5)
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator+(dtype lhs, const NdArray<dtype>& rhs) 
    {
        return rhs + lhs;
    }

    //============================================================================
    // Method Description:
    ///						Adds the scaler to the array (6)
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator+(const NdArray<dtype>& lhs, const std::complex<dtype>& rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](dtype value)  -> std::complex<dtype> 
        {
            return value + rhs;
        };

        NdArray<std::complex<dtype>> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Adds the scaler to the array (7)
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator+(const std::complex<dtype>& lhs, const NdArray<dtype>& rhs) 
    {
        return rhs + lhs;
    }

    //============================================================================
    // Method Description:
    ///						Adds the scaler to the array (8)
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator+(const NdArray<std::complex<dtype>>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](std::complex<dtype> value)  -> std::complex<dtype> 
        {
            return value + rhs;
        };

        NdArray<std::complex<dtype>> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Adds the scaler to the array (9)
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator+(dtype lhs, const NdArray<std::complex<dtype>>& rhs) 
    {
        return rhs + lhs;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the elements of two arrays (1)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator-=(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        stl_algorithms::transform(lhs.begin(), lhs.end(), 
            rhs.cbegin(), lhs.begin(), std::minus<dtype>());

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the elements of two arrays (2)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>>& operator-=(NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        const auto function = [](std::complex<dtype>& val1, dtype val2) -> std::complex<dtype>
        {
            return val1 - val2;
        };

        stl_algorithms::transform(lhs.begin(), lhs.end(), 
            rhs.cbegin(), lhs.begin(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the scaler from the array (3)
    ///
    /// @param      lhs
    ///	@param      rhs
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator-=(NdArray<dtype>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [rhs](dtype& value)  -> dtype
        {
            return value -= rhs;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the scaler from the array (4)
    ///
    /// @param      lhs
    ///	@param      rhs
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>>& operator-=(NdArray<std::complex<dtype>>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](std::complex<dtype>& value)  -> std::complex<dtype>
        {
            return value -= rhs;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the elements of two arrays (1)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator-(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        NdArray<dtype> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), 
            rhs.cbegin(), returnArray.begin(), std::minus<dtype>());

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the elements of two arrays (2)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator-(const NdArray<dtype>& lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        const auto function = [](dtype val1, const std::complex<dtype>& val2) -> std::complex<dtype>
        {
            return val1 - val2;
        };

        NdArray<std::complex<dtype>> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), 
            rhs.cbegin(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the elements of two arrays (3)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator-(const NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        const auto function = [](const std::complex<dtype>&val1, dtype val2) -> std::complex<dtype>
        {
            return val1 - val2;
        };

        NdArray<std::complex<dtype>> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), 
            rhs.cbegin(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the scaler from the array (4)
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator-(const NdArray<dtype>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [rhs](dtype value)  -> dtype 
        {
            return value - rhs;
        };

        NdArray<dtype> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the scaler from the array (5)
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator-(dtype lhs, const NdArray<dtype>& rhs) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [lhs](dtype value)  -> dtype 
        {
            return lhs - value;
        };

        NdArray<dtype> returnArray(rhs.shape());

        stl_algorithms::transform(rhs.cbegin(), rhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the scaler from the array (6)
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator-(const NdArray<dtype>& lhs, const std::complex<dtype>& rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](dtype value)  -> std::complex<dtype> 
        {
            return value - rhs;
        };

        NdArray<std::complex<dtype>> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the scaler from the array (7)
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator-(const std::complex<dtype>& lhs, const NdArray<dtype>& rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [lhs](dtype value)  -> std::complex<dtype> 
        {
            return lhs - value;
        };

        NdArray<std::complex<dtype>> returnArray(rhs.shape());

        stl_algorithms::transform(rhs.cbegin(), rhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the scaler from the array (8)
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator-(const NdArray<std::complex<dtype>>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](std::complex<dtype> value)  -> std::complex<dtype> 
        {
            return value - rhs;
        };

        NdArray<std::complex<dtype>> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the scaler from the array (9)
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator-(dtype lhs, const NdArray<std::complex<dtype>>& rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [lhs](std::complex<dtype> value)  -> std::complex<dtype> 
        {
            return lhs - value;
        };

        NdArray<std::complex<dtype>> returnArray(rhs.shape());

        stl_algorithms::transform(rhs.cbegin(), rhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Negative Operator
    ///
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator-(const NdArray<dtype>& inArray) 
    {
        const auto function = [](dtype value)  -> dtype
        {
            return -value;
        };

        auto returnArray = NdArray<dtype>(inArray.shape());
        stl_algorithms::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), function);
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Multiplies the elements of two arrays (1)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator*=(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        stl_algorithms::transform(lhs.begin(), lhs.end(), 
            rhs.cbegin(), lhs.begin(), std::multiplies<dtype>());

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Multiplies the elements of two arrays (2)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>>& operator*=(NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        const auto function = [](std::complex<dtype>& val1, dtype val2) -> std::complex<dtype>
        {
            return val1 * val2;
        };

        stl_algorithms::transform(lhs.begin(), lhs.end(), 
            rhs.cbegin(), lhs.begin(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Multiplies the scaler to the array (3)
    ///
    /// @param      lhs
    ///	@param      rhs
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator*=(NdArray<dtype>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [rhs](dtype& value)  -> dtype
        {
            return value *= rhs;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Multiplies the scaler to the array (4)
    ///
    /// @param      lhs
    ///	@param      rhs
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>>& operator*=(NdArray<std::complex<dtype>>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](std::complex<dtype>& value)  -> std::complex<dtype>
        {
            return value *= rhs;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Multiplies the elements of two arrays (1)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator*(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        NdArray<dtype> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), 
            rhs.cbegin(), returnArray.begin(), std::multiplies<dtype>());

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Multiplies the elements of two arrays (2)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator*(const NdArray<dtype>& lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        const auto function = [](dtype val1, const std::complex<dtype>& val2) -> std::complex<dtype>
        {
            return val1 * val2;
        };

        NdArray<std::complex<dtype>> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), 
            rhs.cbegin(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Multiplies the elements of two arrays (3)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator*(const NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return rhs * lhs;
    }

    //============================================================================
    // Method Description:
    ///						Multiplies the scaler to the array (4)
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator*(const NdArray<dtype>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [rhs](dtype value)  -> dtype 
        {
            return value * rhs;
        };

        NdArray<dtype> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Multiplies the scaler to the array (5)
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator*(dtype lhs, const NdArray<dtype>& rhs) 
    {
        return rhs * lhs;
    }

    //============================================================================
    // Method Description:
    ///						Multiplies the scaler to the array (6)
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator*(const NdArray<dtype>& lhs, const std::complex<dtype>& rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](dtype value)  -> std::complex<dtype> 
        {
            return value * rhs;
        };

        NdArray<std::complex<dtype>> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Multiplies the scaler to the array (7)
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator*(const std::complex<dtype>& lhs, const NdArray<dtype>& rhs) 
    {
        return rhs * lhs;
    }

    //============================================================================
    // Method Description:
    ///						Multiplies the scaler to the array (8)
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator*(const NdArray<std::complex<dtype>>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](std::complex<dtype> value)  -> std::complex<dtype> 
        {
            return value * rhs;
        };

        NdArray<std::complex<dtype>> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Multiplies the scaler to the array (9)
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator*(dtype lhs, const NdArray<std::complex<dtype>>& rhs) 
    {
        return rhs * lhs;
    }

    //============================================================================
    // Method Description:
    ///						Divides the elements of two arrays (1)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator/=(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        stl_algorithms::transform(lhs.begin(), lhs.end(), 
            rhs.cbegin(), lhs.begin(), std::divides<dtype>());

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Divides the elements of two arrays (2)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>>& operator/=(NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        const auto function = [](std::complex<dtype>& val1, dtype val2) -> std::complex<dtype>
        {
            return val1 / val2;
        };

        stl_algorithms::transform(lhs.begin(), lhs.end(), 
            rhs.cbegin(), lhs.begin(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Divides the scaler from the array (3)
    ///
    /// @param      lhs
    ///	@param      rhs
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator/=(NdArray<dtype>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [rhs](dtype& value)  -> dtype
        {
            return value /= rhs;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Divides the scaler from the array (4)
    ///
    /// @param      lhs
    ///	@param      rhs
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>>& operator/=(NdArray<std::complex<dtype>>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](std::complex<dtype>& value)  -> std::complex<dtype>
        {
            return value /= rhs;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Divides the elements of two arrays (1)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator/(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        NdArray<dtype> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), 
            rhs.cbegin(), returnArray.begin(), std::divides<dtype>());

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Divides the elements of two arrays (2)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator/(const NdArray<dtype>& lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        NdArray<std::complex<dtype>> returnArray(lhs.shape());

        const auto function = [](dtype val1, const std::complex<dtype>& val2) -> std::complex<dtype>
        {
            return val1 / val2;
        };

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), 
            rhs.cbegin(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Divides the elements of two arrays (3)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator/(const NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        const auto function = [](const std::complex<dtype>&val1, dtype val2) -> std::complex<dtype>
        {
            return val1 / val2;
        };

        NdArray<std::complex<dtype>> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), 
            rhs.cbegin(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Divides the scaler from the array (4)
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator/(const NdArray<dtype>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [rhs](dtype value)  -> dtype 
        {
            return value / rhs;
        };

        NdArray<dtype> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Divides the scaler from the array (5)
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator/(dtype lhs, const NdArray<dtype>& rhs) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [lhs](dtype value)  -> dtype 
        {
            return lhs / value;
        };

        NdArray<dtype> returnArray(rhs.shape());

        stl_algorithms::transform(rhs.cbegin(), rhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Divides the scaler from the array (6)
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator/(const NdArray<dtype>& lhs, const std::complex<dtype>& rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](dtype value)  -> std::complex<dtype> 
        {
            return value / rhs;
        };

        NdArray<std::complex<dtype>> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Divides the scaler from the array (7)
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator/(const std::complex<dtype>& lhs, const NdArray<dtype>& rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [lhs](dtype value)  -> std::complex<dtype> 
        {
            return lhs / value;
        };

        NdArray<std::complex<dtype>> returnArray(rhs.shape());

        stl_algorithms::transform(rhs.cbegin(), rhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Divides the scaler from the array (8)
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator/(const NdArray<std::complex<dtype>>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](std::complex<dtype> value)  -> std::complex<dtype> 
        {
            return value / rhs;
        };

        NdArray<std::complex<dtype>> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Divides the scaler from the array (9)
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> operator/(dtype lhs, const NdArray<std::complex<dtype>>& rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [lhs](std::complex<dtype> value)  -> std::complex<dtype> 
        {
            return lhs / value;
        };

        NdArray<std::complex<dtype>> returnArray(rhs.shape());

        stl_algorithms::transform(rhs.cbegin(), rhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Modulus the elements of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator%=(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        stl_algorithms::transform(lhs.begin(), lhs.end(), 
            rhs.cbegin(), lhs.begin(), std::modulus<dtype>());

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Modulus the scaler to the array
    ///
    /// @param      lhs
    ///	@param      rhs
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator%=(NdArray<dtype>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](dtype& value)  -> dtype
        {
            return value %= rhs;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Takes the modulus of the elements of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator%(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        auto returnArray = NdArray<dtype>(lhs);
        returnArray %= rhs;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Modulus of the array and the scaler
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator%(const NdArray<dtype>& lhs, dtype rhs) 
    {
        auto returnArray = NdArray<dtype>(lhs);
        returnArray %= rhs;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Modulus of the scaler and the array
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator%(dtype lhs, const NdArray<dtype>& rhs) 
    {
        NdArray<dtype> returnArray(rhs.shape());
        stl_algorithms::transform(rhs.begin(), rhs.end(), returnArray.begin(),
            [lhs](dtype value)  -> dtype
            {
                return lhs % value;
            });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Bitwise or the elements of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator|=(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_INTEGER(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        stl_algorithms::transform(lhs.begin(), lhs.end(), 
            rhs.cbegin(), lhs.begin(), std::bit_or<dtype>());

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Bitwise or the scaler to the array
    ///
    /// @param      lhs
    ///	@param      rhs
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator|=(NdArray<dtype>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_INTEGER(dtype);

        const auto function = [rhs](dtype& value)  -> dtype
        {
            return value |= rhs;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise or of the elements of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator|(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        auto returnArray = NdArray<dtype>(lhs);
        returnArray |= rhs;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise or of the array and the scaler
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator|(const NdArray<dtype>& lhs, dtype rhs) 
    {
        auto returnArray = NdArray<dtype>(lhs);
        returnArray |= rhs;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise or of the sclar and the array
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator|(dtype lhs, const NdArray<dtype>& rhs) 
    {
        return rhs | lhs;
    }

    //============================================================================
    // Method Description:
    ///						Bitwise and the elements of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator&=(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_INTEGER(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        stl_algorithms::transform(lhs.begin(), lhs.end(), 
            rhs.cbegin(), lhs.begin(), std::bit_and<dtype>());

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Bitwise and the scaler to the array
    ///
    /// @param      lhs
    ///	@param      rhs
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator&=(NdArray<dtype>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_INTEGER(dtype);

        const auto function = [rhs](dtype& value)  -> dtype
        {
            return value &= rhs;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise and of the elements of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator&(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        auto returnArray = NdArray<dtype>(lhs);
        returnArray &= rhs;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise and of the array and the scaler
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator&(const NdArray<dtype>& lhs, dtype rhs) 
    {
        auto returnArray = NdArray<dtype>(lhs);
        returnArray &= rhs;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise and of the sclar and the array
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator&(dtype lhs, const NdArray<dtype>& rhs) 
    {
        return rhs & lhs;
    }

    //============================================================================
    // Method Description:
    ///						Bitwise xor the elements of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator^=(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_INTEGER(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        stl_algorithms::transform(lhs.begin(), lhs.end(), 
            rhs.cbegin(), lhs.begin(), std::bit_xor<dtype>());

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Bitwise xor the scaler to the array
    ///
    /// @param      lhs
    ///	@param      rhs
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator^=(NdArray<dtype>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_INTEGER(dtype);

        const auto function = [rhs](dtype& value)  -> dtype
        {
            return value ^= rhs;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise xor of the elements of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator^(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        auto returnArray = NdArray<dtype>(lhs);
        returnArray ^= rhs;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise xor of the array and the scaler
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator^(const NdArray<dtype>& lhs, dtype rhs) 
    {
        auto returnArray = NdArray<dtype>(lhs);
        returnArray ^= rhs;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise xor of the sclar and the array
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator^(dtype lhs, const NdArray<dtype>& rhs) 
    {
        return rhs ^ lhs;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise not of the array
    ///
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator~(const NdArray<dtype>& inArray) 
    {
        STATIC_ASSERT_INTEGER(dtype);

        const auto function = [](dtype value)  -> dtype
        {
            return ~value;
        };

        NdArray<dtype> returnArray(inArray.shape());

        stl_algorithms::transform(inArray.cbegin(), inArray.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the and of the elements of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator&&(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        const auto function = [](dtype value1, dtype value2)  -> bool
        {
            return value1 && value2;
        };

        NdArray<bool> returnArray(lhs.shape());
        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), 
            rhs.cbegin(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the and of the array and the scaler
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator&&(const NdArray<dtype>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        NdArray<bool> returnArray(lhs.shape());

        const auto function = [rhs](dtype value)  -> bool
        {
            return value && rhs;
        };

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the and of the array and the scaler
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator&&(dtype lhs, const NdArray<dtype>& rhs) 
    {
        return rhs && lhs;
    }

    //============================================================================
    // Method Description:
    ///						Takes the or of the elements of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator||(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        const auto function = [](dtype value1, dtype value2)  -> bool
        {
            return value1 || value2;
        };

        NdArray<bool> returnArray(lhs.shape());
        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), 
            rhs.cbegin(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the or of the array and the scaler
    ///
    /// @param      lhs
    ///	@param  	rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator||(const NdArray<dtype>& lhs, dtype rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        NdArray<bool> returnArray(lhs.shape());

        const auto function = [rhs](dtype value)  -> bool
        {
            return value || rhs;
        };

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the or of the array and the scaler
    ///
    ///	@param  	lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator||(dtype lhs, const NdArray<dtype>& rhs) 
    {
        return rhs || lhs;
    }

    //============================================================================
    // Method Description:
    ///						Takes the not of the array
    ///
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator!(const NdArray<dtype>& inArray) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        NdArray<bool> returnArray(inArray.shape());

        const auto function = [](dtype value)  -> dtype
        {
            return !value;
        };

        stl_algorithms::transform(inArray.cbegin(), inArray.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator==(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        const auto equalTo = [](dtype lhs, dtype rhs) noexcept -> bool
        {
            return utils::essentiallyEqual(lhs, rhs);
        };

        NdArray<bool> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), 
            rhs.cbegin(), returnArray.begin(), equalTo);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						an array and a scaler
    ///
    /// @param      lhs
    ///	@param  	inValue
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator==(const NdArray<dtype>& lhs, dtype inValue) 
    {
        NdArray<bool> returnArray(lhs.shape());

        const auto equalTo = [inValue](dtype value) noexcept -> bool
        {
            return utils::essentiallyEqual(inValue, value);
        };

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), 
            returnArray.begin(), equalTo);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						an array and a scaler
    ///
    ///	@param  	inValue
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator==(dtype inValue, const NdArray<dtype>& inArray) 
    {
        return inArray == inValue;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator!=(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        const auto notEqualTo = [](dtype lhs, dtype rhs) noexcept -> bool
        {
            return !utils::essentiallyEqual(lhs, rhs);
        };

        NdArray<bool> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), 
            rhs.cbegin(), returnArray.begin(), notEqualTo);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						an array and a scaler
    ///
    /// @param      lhs
    ///	@param  	inValue
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator!=(const NdArray<dtype>& lhs, dtype inValue) 
    {
        NdArray<bool> returnArray(lhs.shape());

        const auto notEqualTo = [inValue](dtype value) noexcept -> bool
        {
            return !utils::essentiallyEqual(inValue, value);
        };

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), 
            returnArray.begin(), notEqualTo);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						an array and a scaler
    ///
    ///	@param  	inValue
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator!=(dtype inValue, const NdArray<dtype>& inArray) 
    {
        return inArray != inValue;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator<(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        NdArray<bool> returnArray(lhs.shape());

        const auto function = [](dtype lhs, dtype rhs) noexcept -> bool
        {
            return lhs < rhs;
        };

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(),
            rhs.cbegin(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						the array and a scaler
    ///
    /// @param      lhs
    ///	@param  	inValue
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator<(const NdArray<dtype>& lhs, dtype inValue) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        NdArray<bool> returnArray(lhs.shape());

        const auto function = [inValue](dtype value) noexcept -> bool
        {
            return value < inValue;
        };

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						the array and a scaler
    ///
    ///	@param  	inValue
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator<(dtype inValue, const NdArray<dtype>& inArray) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        NdArray<bool> returnArray(inArray.shape());

        const auto function = [inValue](dtype value) noexcept -> bool
        {
            return inValue < value;
        };

        stl_algorithms::transform(inArray.cbegin(), inArray.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator>(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        NdArray<bool> returnArray(lhs.shape());

        const auto function = [](dtype lhs, dtype rhs) noexcept -> bool
        {
            return lhs > rhs;
        };

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(),
            rhs.cbegin(), returnArray.begin(), function);

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(),
            rhs.cbegin(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						the array and a scaler
    ///
    /// @param      lhs
    ///	@param  	inValue
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator>(const NdArray<dtype>& lhs, dtype inValue) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        NdArray<bool> returnArray(lhs.shape());

        const auto function = [inValue](dtype value) noexcept -> bool
        {
            return value > inValue;
        };

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						the array and a scaler
    ///
    ///	@param  	inValue
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator>(dtype inValue, const NdArray<dtype>& inArray) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        NdArray<bool> returnArray(inArray.shape());

        const auto function = [inValue](dtype value) noexcept -> bool
        {
            return inValue > value;
        };

        stl_algorithms::transform(inArray.cbegin(), inArray.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator<=(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        NdArray<bool> returnArray(lhs.shape());

        const auto function = [](dtype lhs, dtype rhs) noexcept -> bool
        {
            return lhs <= rhs;
        };

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(),
            rhs.cbegin(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						the array and a scaler
    ///
    /// @param      lhs
    ///	@param  	inValue
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator<=(const NdArray<dtype>& lhs, dtype inValue) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        NdArray<bool> returnArray(lhs.shape());

        const auto function = [inValue](dtype value) noexcept -> bool
        {
            return value <= inValue;
        };

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						the array and a scaler
    ///
    ///	@param  	inValue
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator<=(dtype inValue, const NdArray<dtype>& inArray) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        NdArray<bool> returnArray(inArray.shape());

        const auto function = [inValue](dtype value) noexcept -> bool
        {
            return inValue <= value;
        };

        stl_algorithms::transform(inArray.cbegin(), inArray.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator>=(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (lhs.shape() != rhs.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        NdArray<bool> returnArray(lhs.shape());

        const auto function = [](dtype lhs, dtype rhs) noexcept -> bool
        {
            return lhs >= rhs;
        };

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(),
            rhs.cbegin(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						the array and a scaler
    ///
    /// @param      lhs
    ///	@param  	inValue
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator>=(const NdArray<dtype>& lhs, dtype inValue) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        NdArray<bool> returnArray(lhs.shape());

        const auto function = [inValue](dtype value) noexcept -> bool
        {
            return value >= inValue;
        };

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						the array and a scaler
    ///
    ///	@param  	inValue
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator>=(dtype inValue, const NdArray<dtype>& inArray) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        NdArray<bool> returnArray(inArray.shape());

        const auto function = [inValue](dtype value) noexcept -> bool
        {
            return inValue >= value;
        };

        stl_algorithms::transform(inArray.cbegin(), inArray.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Bitshifts left the elements of the array
    ///
    /// @param      lhs
    /// @param      inNumBits
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator<<=(NdArray<dtype>& lhs, uint8 inNumBits) 
    {
        STATIC_ASSERT_INTEGER(dtype);

        const auto function = [inNumBits](dtype& value)  -> void
        {
            value <<= inNumBits;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Bitshifts left the elements of the array
    ///
    /// @param      lhs
    /// @param      inNumBits
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator<<(const NdArray<dtype>& lhs, uint8 inNumBits) 
    {
        STATIC_ASSERT_INTEGER(dtype);

        NdArray<dtype> returnArray(lhs);
        returnArray <<= inNumBits;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Bitshifts right the elements of the array
    ///
    /// @param      lhs
    /// @param      inNumBits
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator>>=(NdArray<dtype>& lhs, uint8 inNumBits) 
    {
        STATIC_ASSERT_INTEGER(dtype);

        const auto function = [inNumBits](dtype& value)  -> void
        {
            value >>= inNumBits;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Bitshifts right the elements of the array
    ///       
    /// @param      lhs
    /// @param      inNumBits
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator>>(const NdArray<dtype>& lhs, uint8 inNumBits) 
    {
        STATIC_ASSERT_INTEGER(dtype);

        NdArray<dtype> returnArray(lhs);
        returnArray >>= inNumBits;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						prefix incraments the elements of an array
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator++(NdArray<dtype>& rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [](dtype& value)  -> void
        {
            ++value;
        };

        stl_algorithms::for_each(rhs.begin(), rhs.end(), function);

        return rhs;
    }

    //============================================================================
    // Method Description:
    ///						postfix increments the elements of an array
    ///
    /// @param      lhs
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator++(NdArray<dtype>& lhs, int) 
    {
        auto copy = NdArray<dtype>(lhs);
        ++lhs;
        return copy;
    }

    //============================================================================
    // Method Description:
    ///						prefix decrements the elements of an array
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator--(NdArray<dtype>& rhs) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [](dtype& value)  -> void
        {
            --value;
        };

        stl_algorithms::for_each(rhs.begin(), rhs.end(), function);

        return rhs;
    }

    //============================================================================
    // Method Description:
    ///						postfix decrements the elements of an array
    ///
    /// @param      lhs
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator--(NdArray<dtype>& lhs, int) 
    {
        auto copy = NdArray<dtype>(lhs);
        --lhs;
        return copy;
    }

    //============================================================================
    // Method Description:
    ///						io operator for the NdArray class
    ///
    /// @param      inOStream
    /// @param      inArray
    /// @return
    ///				std::ostream
    ///
    template<typename dtype>
    std::ostream& operator<<(std::ostream& inOStream, const NdArray<dtype>& inArray) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        inOStream << inArray.str();
        return inOStream;
    }
}  // namespace nc
