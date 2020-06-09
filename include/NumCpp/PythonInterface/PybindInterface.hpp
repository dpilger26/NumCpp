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
/// A module for interacting with python with pybind11 interface
///
#pragma once

#ifdef INCLUDE_PYBIND_PYTHON_INTERFACE

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/NdArray.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <map>
#include <utility>

namespace nc
{
    namespace pybindInterface
    {
        /// Enum for the pybind array return policy
        enum class ReturnPolicy { COPY, REFERENCE, TAKE_OWNERSHIP };

        static const std::map<ReturnPolicy, std::string> returnPolicyStringMap = { {ReturnPolicy::COPY, "COPY"},
        {ReturnPolicy::REFERENCE, "REFERENCE"},
        {ReturnPolicy::TAKE_OWNERSHIP, "TAKE_OWNERSHIP"} };

        //============================================================================
        ///						converts a numpy array to a numcpp NdArray using pybind bindings
        ///                     Python will still own the underlying data.
        ///
        /// @param      numpyArray
        ///
        /// @return     NdArray<dtype, Alloc>
        ///
        template<typename dtype, class Alloc = std::allocator<dtype>>
        inline NdArray<dtype, Alloc> pybind2nc(pybind11::array_t<dtype, pybind11::array::c_style>& numpyArray)
        {
            dtype* dataPtr = numpyArray.mutable_data();
            switch (numpyArray.ndim())
            {
                case 0:
                {
                    return NdArray<dtype, Alloc>(dataPtr, 0, 0, false);
                }
                case 1:
                {
                    uint32 size = static_cast<uint32>(numpyArray.size());
                    return NdArray<dtype, Alloc>(dataPtr, 1, size, false);
                }
                case 2:
                {
                    uint32 numRows = static_cast<uint32>(numpyArray.shape(0));
                    uint32 numCols = static_cast<uint32>(numpyArray.shape(1));
                    return NdArray<dtype, Alloc>(dataPtr, numRows, numCols, false);
                }
                default:
                {
                    THROW_INVALID_ARGUMENT_ERROR("input array must be no more than 2 dimensional.");
                }
            }
        }

        //============================================================================
        ///						converts a numcpp NdArray to numpy array using pybind bindings
        ///
        /// @param     inArray: the input array
        /// @param     returnPolicy: the return policy
        ///
        /// @return    pybind11::array_t
        ///
        template<typename dtype, class Alloc>
        inline pybind11::array_t<dtype> nc2pybind(NdArray<dtype, Alloc>& inArray, 
            ReturnPolicy returnPolicy = ReturnPolicy::COPY)
        {
            Shape inShape = inArray.shape();
            std::vector<pybind11::ssize_t> shape{ inShape.rows, inShape.cols };
            std::vector<pybind11::ssize_t> strides{ inShape.cols * sizeof(dtype), sizeof(dtype) };

            switch (returnPolicy)
            {
                case ReturnPolicy::COPY:
                {
                    return pybind11::array_t<dtype>(shape, strides, inArray.data());
                }
                case ReturnPolicy::REFERENCE:
                {
                    typename pybind11::capsule reference(inArray.data(), [](void* ptr) {});
                    return pybind11::array_t<dtype>(shape, strides, inArray.data(), reference);
                }
                case ReturnPolicy::TAKE_OWNERSHIP:
                {
                    typename pybind11::capsule garbageCollect(inArray.dataRelease(),
                        [](void* ptr)
                        {
                            dtype* dataPtr = reinterpret_cast<dtype*>(ptr);
                            delete[] dataPtr;
                        }
                    );
                    return pybind11::array_t<dtype>(shape, strides, inArray.data(), garbageCollect);
                }
                default:
                {
                    std::stringstream sstream;
                    sstream << "ReturnPolicy " << returnPolicyStringMap.at(returnPolicy) << " has not been implemented yet" << std::endl;
                    THROW_INVALID_ARGUMENT_ERROR(sstream.str());
                }
            }
        }
    }
}
#endif
