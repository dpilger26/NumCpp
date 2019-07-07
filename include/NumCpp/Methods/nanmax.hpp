/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.0
///
/// @section License
/// Copyright 2019 David Pilger
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
/// Methods for working with NdArrays
///
#pragma once

#include"NumCpp/Core/DtypeInfo.hpp"
#include"NumCpp/Core/Types.hpp"
#include"NumCpp/Methods/max.hpp"
#include"NumCpp/NdArray/NdArray.hpp"

#include<algorithm>
#include<cmath>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Return the maximum of an array or maximum along an axis ignoring NaNs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nanmax.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> nanmax(const NdArray<dtype>& inArray, Axis inAxis) noexcept
    {
        NdArray<dtype> arrayCopy(inArray);
        std::for_each(arrayCopy.begin(), arrayCopy.end(),
            [](dtype& value) noexcept -> void 
            { if (std::isnan(value)) { value = DtypeInfo<dtype>::min(); }; });

        return max(arrayCopy, inAxis);
    }
}
