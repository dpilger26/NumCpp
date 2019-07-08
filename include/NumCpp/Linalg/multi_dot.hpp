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
/// Compute the dot product of two or more arrays in a single function call.
///
#pragma once

#include "NumCpp/Methods/dot.hpp"
#include "NumCpp/NdArray.hpp"

#include <iostream>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace nc
{
    namespace linalg
    {
        //============================================================================
        // Method Description:
        ///						Compute the dot product of two or more arrays in a single
        ///						function call.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.multi_dot.html#numpy.linalg.multi_dot
        ///
        /// @param
        ///				inList: list of arrays
        ///
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> multi_dot(const std::initializer_list<NdArray<dtype> >& inList)
        {
            typename std::initializer_list<NdArray<dtype> >::iterator iter = inList.begin();

            if (inList.size() == 0)
            {
                std::string errStr = "ERROR: linalg::multi_dot: input empty list of arrays.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }
            else if (inList.size() == 1)
            {
                return iter->copy();
            }

            NdArray<dtype> returnArray = dot<dtype>(*iter, *(iter + 1));
            iter += 2;
            for (; iter < inList.end(); ++iter)
            {
                returnArray = dot(returnArray, *iter);
            }

            return returnArray;
        }
    }
}
