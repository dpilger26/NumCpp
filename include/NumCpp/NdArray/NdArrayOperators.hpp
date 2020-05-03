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
/// Holds 1D and 2D arrays, the main work horse of the NumCpp library
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/NdArray/NdArrayCore.hpp"

#include <algorithm>
#include <functional>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Adds the elements of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator+=(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
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
    ///						Adds the scalar to the array
    ///
    /// @param      lhs
    ///	@param      inScalar
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator+=(NdArray<dtype>& lhs, dtype inScalar) noexcept
    {
        auto function = [inScalar](dtype& value) noexcept -> dtype
        {
            return value += inScalar;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Adds the elements of two arrays
    ///
    /// @param      inArray1
    /// @param      inArray2
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator+(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        auto returnArray = NdArray<dtype>(inArray1);
        returnArray += inArray2;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Adds the scalar to the array
    ///
    /// @param      inArray
    ///	@param  	inScalar
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator+(const NdArray<dtype>& inArray, dtype inScalar) noexcept
    {
        auto returnArray = NdArray<dtype>(inArray);
        returnArray += inScalar;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Adds the scalar to the array
    ///
    ///	@param  	inScalar
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator+(dtype inScalar, const NdArray<dtype>& inArray) noexcept
    {
        return inArray + inScalar;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the elements of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator-=(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
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
    ///						Subtracts the scalar from the array
    ///
    /// @param      lhs
    ///	@param      inScalar
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator-=(NdArray<dtype>& lhs, dtype inScalar) noexcept
    {
        auto function = [inScalar](dtype& value) noexcept -> dtype
        {
            return value -= inScalar;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the elements of two arrays
    ///
    /// @param      inArray1
    /// @param      inArray2
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator-(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        auto returnArray = NdArray<dtype>(inArray1);
        returnArray -= inArray2;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the scalar from the array
    ///
    /// @param      inArray
    ///	@param  	inScalar
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator-(const NdArray<dtype>& inArray, dtype inScalar) noexcept
    {
        auto returnArray = NdArray<dtype>(inArray);
        returnArray -= inScalar;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the array from the scalar
    ///
    ///	@param  	inScalar
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator-(dtype inScalar, const NdArray<dtype>& inArray) noexcept
    {
        auto returnArray = -inArray;
        returnArray += inScalar;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Negative Operator
    ///
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator-(const NdArray<dtype>& inArray) noexcept
    {
        auto returnArray = NdArray<dtype>(inArray);
        returnArray *= static_cast<dtype>(-1);
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Multiples the elements of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator*=(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
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
    ///						Multiples the scalar to the array
    ///
    /// @param      lhs
    ///	@param      inScalar
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator*=(NdArray<dtype>& lhs, dtype inScalar) noexcept
    {
        auto function = [inScalar](dtype& value) noexcept -> dtype
        {
            return value *= inScalar;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Multiplies the elements of two arrays
    ///
    /// @param      inArray1
    /// @param      inArray2
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator*(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        auto returnArray = NdArray<dtype>(inArray1);
        returnArray *= inArray2;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Multiplies the scalar to the array
    ///
    /// @param      inArray
    ///	@param  	inScalar
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator*(const NdArray<dtype>& inArray, dtype inScalar) noexcept
    {
        auto returnArray = NdArray<dtype>(inArray);
        returnArray *= inScalar;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Multiplies the scalar to the array
    ///
    ///	@param  	inScalar
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator*(dtype inScalar, const NdArray<dtype>& inArray) noexcept
    {
        return inArray * inScalar;
    }

    //============================================================================
    // Method Description:
    ///						Divides the elements of two arrays
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator/=(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
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
    ///						Multiples the scalar to the array
    ///
    /// @param      lhs
    ///	@param      inScalar
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator/=(NdArray<dtype>& lhs, dtype inScalar) noexcept
    {
        auto function = [inScalar](dtype& value) noexcept -> dtype
        {
            return value /= inScalar;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Divides the elements of two arrays
    ///
    /// @param      inArray1
    /// @param      inArray2
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator/(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        auto returnArray = NdArray<dtype>(inArray1);
        returnArray /= inArray2;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Divides the scalar by the array
    ///
    /// @param      inArray
    ///	@param  	inScalar
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator/(const NdArray<dtype>& inArray, dtype inScalar) noexcept
    {
        auto returnArray = NdArray<dtype>(inArray);
        returnArray /= inScalar;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Divides the array by the scalar
    ///
    ///	@param  	inScalar
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator/(dtype inScalar, const NdArray<dtype>& inArray) noexcept
    {
        NdArray<dtype> returnArray(inArray.shape());
        stl_algorithms::transform(inArray.begin(), inArray.end(), returnArray.begin(),
            [inScalar](dtype value) noexcept -> dtype
            {
                return inScalar / value;
            });

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
    ///						Modulus the scalar to the array
    ///
    /// @param      lhs
    ///	@param      inScalar
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator%=(NdArray<dtype>& lhs, dtype inScalar) noexcept
    {
        auto function = [inScalar](dtype& value) noexcept -> dtype
        {
            return value %= inScalar;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Takes the modulus of the elements of two arrays
    ///
    /// @param      inArray1
    /// @param      inArray2
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator%(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        auto returnArray = NdArray<dtype>(inArray1);
        returnArray %= inArray2;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Modulus of the array and the scalar
    ///
    /// @param      inArray
    ///	@param  	inScalar
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator%(const NdArray<dtype>& inArray, dtype inScalar) noexcept
    {
        auto returnArray = NdArray<dtype>(inArray);
        returnArray %= inScalar;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Modulus of the scalar and the array
    ///
    ///	@param  	inScalar
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator%(dtype inScalar, const NdArray<dtype>& inArray) noexcept
    {
        NdArray<dtype> returnArray(inArray.shape());
        stl_algorithms::transform(inArray.begin(), inArray.end(), returnArray.begin(),
            [inScalar](dtype value) noexcept -> dtype
            {
                return inScalar % value;
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
    ///						Bitwise or the scalar to the array
    ///
    /// @param      lhs
    ///	@param      inScalar
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator|=(NdArray<dtype>& lhs, dtype inScalar) noexcept
    {
        auto function = [inScalar](dtype& value) noexcept -> dtype
        {
            return value |= inScalar;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise or of the elements of two arrays
    ///
    /// @param      inArray1
    /// @param      inArray2
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator|(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        auto returnArray = NdArray<dtype>(inArray1);
        returnArray |= inArray2;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise or of the array and the scalar
    ///
    /// @param      inArray
    ///	@param  	inScalar
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator|(const NdArray<dtype>& inArray, dtype inScalar) noexcept
    {
        auto returnArray = NdArray<dtype>(inArray);
        returnArray |= inScalar;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise or of the sclar and the array
    ///
    ///	@param  	inScalar
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator|(dtype inScalar, const NdArray<dtype>& inArray) noexcept
    {
        return inArray | inScalar;
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
    ///						Bitwise and the scalar to the array
    ///
    /// @param      lhs
    ///	@param      inScalar
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator&=(NdArray<dtype>& lhs, dtype inScalar) noexcept
    {
        auto function = [inScalar](dtype& value) noexcept -> dtype
        {
            return value &= inScalar;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise and of the elements of two arrays
    ///
    /// @param      inArray1
    /// @param      inArray2
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator&(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        auto returnArray = NdArray<dtype>(inArray1);
        returnArray &= inArray2;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise and of the array and the scalar
    ///
    /// @param      inArray
    ///	@param  	inScalar
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator&(const NdArray<dtype>& inArray, dtype inScalar) noexcept
    {
        auto returnArray = NdArray<dtype>(inArray);
        returnArray &= inScalar;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise and of the sclar and the array
    ///
    ///	@param  	inScalar
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator&(dtype inScalar, const NdArray<dtype>& inArray) noexcept
    {
        return inArray & inScalar;
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
    ///						Bitwise xor the scalar to the array
    ///
    /// @param      lhs
    ///	@param      inScalar
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator^=(NdArray<dtype>& lhs, dtype inScalar) noexcept
    {
        auto function = [inScalar](dtype& value) noexcept -> dtype
        {
            return value ^= inScalar;
        };

        stl_algorithms::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise xor of the elements of two arrays
    ///
    /// @param      inArray1
    /// @param      inArray2
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator^(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        auto returnArray = NdArray<dtype>(inArray1);
        returnArray ^= inArray2;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise xor of the array and the scalar
    ///
    /// @param      inArray
    ///	@param  	inScalar
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator^(const NdArray<dtype>& inArray, dtype inScalar) noexcept
    {
        auto returnArray = NdArray<dtype>(inArray);
        returnArray ^= inScalar;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise xor of the sclar and the array
    ///
    ///	@param  	inScalar
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator^(dtype inScalar, const NdArray<dtype>& inArray) noexcept
    {
        return inArray ^ inScalar;
    }

    //============================================================================
    // Method Description:
    ///						Takes the bitwise not of the array
    ///
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> operator~(const NdArray<dtype>& inArray) noexcept
    {
        STATIC_ASSERT_INTEGER(dtype);

        auto function = [](dtype value) noexcept -> dtype
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
    /// @param      inArray1
    /// @param      inArray2
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator&&(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        auto function = [](dtype value1, dtype value2) noexcept -> bool
        {
            return value1 && value2;
        };

        NdArray<bool> returnArray(inArray1.shape());
        stl_algorithms::transform(inArray1.cbegin(), inArray1.cend(), 
            inArray2.cbegin(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the and of the array and the scalar
    ///
    /// @param      inArray
    ///	@param  	inScalar
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator&&(const NdArray<dtype>& inArray, dtype inScalar) noexcept
    {
        NdArray<bool> returnArray(inArray.shape());

        auto function = [inScalar](dtype value) noexcept -> bool
        {
            return value && inScalar;
        };

        stl_algorithms::transform(inArray.cbegin(), inArray.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the and of the array and the scalar
    ///
    ///	@param  	inScalar
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator&&(dtype inScalar, const NdArray<dtype>& inArray) noexcept
    {
        return inArray && inScalar;
    }

    //============================================================================
    // Method Description:
    ///						Takes the or of the elements of two arrays
    ///
    /// @param      inArray1
    /// @param      inArray2
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator||(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        auto function = [](dtype value1, dtype value2) noexcept -> bool
        {
            return value1 || value2;
        };

        NdArray<bool> returnArray(inArray1.shape());
        stl_algorithms::transform(inArray1.cbegin(), inArray1.cend(), 
            inArray2.cbegin(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the or of the array and the scalar
    ///
    /// @param      inArray
    ///	@param  	inScalar
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator||(const NdArray<dtype>& inArray, dtype inScalar) noexcept
    {
        NdArray<bool> returnArray(inArray.shape());

        auto function = [inScalar](dtype value) noexcept -> bool
        {
            return value || inScalar;
        };

        stl_algorithms::transform(inArray.cbegin(), inArray.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Takes the or of the array and the scalar
    ///
    ///	@param  	inScalar
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator||(dtype inScalar, const NdArray<dtype>& inArray) noexcept
    {
        return inArray || inScalar;
    }

    //============================================================================
    // Method Description:
    ///						Takes the not of the array
    ///
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator!(const NdArray<dtype>& inArray) noexcept
    {
        NdArray<bool> returnArray(inArray.shape());

        auto function = [](dtype value) noexcept -> dtype
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
    /// @param      inArray1
    /// @param      inArray2
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator==(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        NdArray<bool> returnArray(inArray1.shape());

        stl_algorithms::transform(inArray1.cbegin(), inArray1.cend(), 
            inArray2.cbegin(), returnArray.begin(), std::equal_to<dtype>());

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						an array and a scalar
    ///
    /// @param      inArray
    ///	@param  	inValue
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator==(const NdArray<dtype>& inArray, dtype inValue) noexcept
    {
        NdArray<bool> returnArray(inArray.shape());

        auto function = [inValue](dtype value) noexcept -> bool
        {
            return value == inValue;
        };

        stl_algorithms::transform(inArray.cbegin(), inArray.cend(), 
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						an array and a scalar
    ///
    ///	@param  	inValue
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator==(dtype inValue, const NdArray<dtype>& inArray) noexcept
    {
        return inArray == inValue;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						of two arrays
    ///
    /// @param      inArray1
    /// @param      inArray2
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator!=(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        NdArray<bool> returnArray(inArray1.shape());

        stl_algorithms::transform(inArray1.cbegin(), inArray1.cend(), 
            inArray2.cbegin(), returnArray.begin(), std::not_equal_to<dtype>());

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						an array and a scalar
    ///
    /// @param      inArray
    ///	@param  	inValue
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator!=(const NdArray<dtype>& inArray, dtype inValue) noexcept
    {
        NdArray<bool> returnArray(inArray.shape());

        auto function = [inValue](dtype value) noexcept -> bool
        {
            return value != inValue;
        };

        stl_algorithms::transform(inArray.cbegin(), inArray.cend(), 
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						an array and a scalar
    ///
    ///	@param  	inValue
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator!=(dtype inValue, const NdArray<dtype>& inArray) noexcept
    {
        return inArray != inValue;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						of two arrays
    ///
    /// @param      inArray1
    /// @param      inArray2
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator<(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        NdArray<bool> returnArray(inArray1.shape());

        stl_algorithms::transform(inArray1.cbegin(), inArray1.cend(),
            inArray2.cbegin(), returnArray.begin(), std::less<dtype>());

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						the array and a scalar
    ///
    /// @param      inArray
    ///	@param  	inValue
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator<(const NdArray<dtype>& inArray, dtype inValue) noexcept
    {
        NdArray<bool> returnArray(inArray.shape());

        auto function = [inValue](dtype value) noexcept -> bool
        {
            return value < inValue;
        };

        stl_algorithms::transform(inArray.cbegin(), inArray.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						the array and a scalar
    ///
    ///	@param  	inValue
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator<(dtype inValue, const NdArray<dtype>& inArray) noexcept
    {
        NdArray<bool> returnArray(inArray.shape());

        auto function = [inValue](dtype value) noexcept -> bool
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
    /// @param      inArray1
    /// @param      inArray2
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator>(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        NdArray<bool> returnArray(inArray1.shape());

        stl_algorithms::transform(inArray1.cbegin(), inArray1.cend(),
            inArray2.cbegin(), returnArray.begin(), std::greater<dtype>());

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						the array and a scalar
    ///
    /// @param      inArray
    ///	@param  	inValue
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator>(const NdArray<dtype>& inArray, dtype inValue) noexcept
    {
        NdArray<bool> returnArray(inArray.shape());

        auto function = [inValue](dtype value) noexcept -> bool
        {
            return value > inValue;
        };

        stl_algorithms::transform(inArray.cbegin(), inArray.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						the array and a scalar
    ///
    ///	@param  	inValue
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator>(dtype inValue, const NdArray<dtype>& inArray) noexcept
    {
        NdArray<bool> returnArray(inArray.shape());

        auto function = [inValue](dtype value) noexcept -> bool
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
    /// @param      inArray1
    /// @param      inArray2
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator<=(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        NdArray<bool> returnArray(inArray1.shape());

        stl_algorithms::transform(inArray1.cbegin(), inArray1.cend(),
            inArray2.cbegin(), returnArray.begin(), std::less_equal<dtype>());

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						the array and a scalar
    ///
    /// @param      inArray
    ///	@param  	inValue
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator<=(const NdArray<dtype>& inArray, dtype inValue) noexcept
    {
        NdArray<bool> returnArray(inArray.shape());

        auto function = [inValue](dtype value) noexcept -> bool
        {
            return value <= inValue;
        };

        stl_algorithms::transform(inArray.cbegin(), inArray.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						the array and a scalar
    ///
    ///	@param  	inValue
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator<=(dtype inValue, const NdArray<dtype>& inArray) noexcept
    {
        NdArray<bool> returnArray(inArray.shape());

        auto function = [inValue](dtype value) noexcept -> bool
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
    /// @param      inArray1
    /// @param      inArray2
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator>=(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Array dimensions do not match.");
        }

        NdArray<bool> returnArray(inArray1.shape());

        stl_algorithms::transform(inArray1.cbegin(), inArray1.cend(),
            inArray2.cbegin(), returnArray.begin(), std::greater_equal<dtype>());

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						the array and a scalar
    ///
    /// @param      inArray
    ///	@param  	inValue
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator>=(const NdArray<dtype>& inArray, dtype inValue) noexcept
    {
        NdArray<bool> returnArray(inArray.shape());

        auto function = [inValue](dtype value) noexcept -> bool
        {
            return value >= inValue;
        };

        stl_algorithms::transform(inArray.cbegin(), inArray.cend(),
            returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns an array of booleans of element wise comparison
    ///						the array and a scalar
    ///
    ///	@param  	inValue
    /// @param      inArray
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<bool> operator>=(dtype inValue, const NdArray<dtype>& inArray) noexcept
    {
        NdArray<bool> returnArray(inArray.shape());

        auto function = [inValue](dtype value) noexcept -> bool
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
    /// @param      inArray
    /// @param      inNumBits
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator<<=(NdArray<dtype>& inArray, uint8 inNumBits) noexcept
    {
        STATIC_ASSERT_INTEGER(dtype);

        auto function = [inNumBits](dtype& value) noexcept -> void
        {
            value <<= inNumBits;
        };

        stl_algorithms::for_each(inArray.begin(), inArray.end(), function);

        return inArray;
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
    NdArray<dtype> operator<<(const NdArray<dtype>& lhs, uint8 inNumBits) noexcept
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
    /// @param      inArray
    /// @param      inNumBits
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& operator>>=(NdArray<dtype>& inArray, uint8 inNumBits) noexcept
    {
        STATIC_ASSERT_INTEGER(dtype);

        auto function = [inNumBits](dtype& value) noexcept -> void
        {
            value >>= inNumBits;
        };

        stl_algorithms::for_each(inArray.begin(), inArray.end(), function);

        return inArray;
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
    NdArray<dtype> operator>>(const NdArray<dtype>& lhs, uint8 inNumBits) noexcept
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
    NdArray<dtype>& operator++(NdArray<dtype>& rhs) noexcept
    {
        auto function = [](dtype& value) noexcept -> void
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
    NdArray<dtype> operator++(NdArray<dtype>& lhs, int) noexcept
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
    NdArray<dtype>& operator--(NdArray<dtype>& rhs) noexcept
    {
        auto function = [](dtype& value) noexcept -> void
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
    NdArray<dtype> operator--(NdArray<dtype>& lhs, int) noexcept
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
    std::ostream& operator<<(std::ostream& inOStream, const NdArray<dtype>& inArray) noexcept
    {
        inOStream << inArray.str();
        return inOStream;
    }
}
