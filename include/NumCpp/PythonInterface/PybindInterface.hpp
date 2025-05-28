/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2025 David Pilger
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
/// A module for interacting with python with pybind11 interface
///
#pragma once

#ifdef NUMCPP_INCLUDE_PYBIND_PYTHON_INTERFACE

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include <map>
#include <utility>

#include "NumCpp/Core/Enums.hpp"
#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::pybindInterface
{
    /// Enum for the pybind array return policy
    enum class ReturnPolicy
    {
        COPY,
        REFERENCE,
        TAKE_OWNERSHIP
    };

    static const std::map<ReturnPolicy, std::string> returnPolicyStringMap = { { ReturnPolicy::COPY, "COPY" },
                                                                               { ReturnPolicy::REFERENCE, "REFERENCE" },
                                                                               { ReturnPolicy::TAKE_OWNERSHIP,
                                                                                 "TAKE_OWNERSHIP" } };

    template<typename dtype>
    using pbArray        = pybind11::array_t<dtype, pybind11::array::c_style>;
    using pbArrayGeneric = pybind11::array;

    //============================================================================
    /// converts a numpy array to a numcpp NdArray using pybind bindings
    /// Python will still own the underlying data.
    ///
    /// @param numpyArray
    ///
    /// @return NdArray<dtype>
    ///
    template<typename dtype>
    NdArray<dtype> pybind2nc(pbArray<dtype>& numpyArray)
    {
        const auto dataPtr = numpyArray.mutable_data();
        switch (numpyArray.ndim())
        {
            case 0:
            {
                return NdArray<dtype>(dataPtr, 0, 0, PointerPolicy::COPY);
            }
            case 1:
            {
                const auto size = static_cast<uint32>(numpyArray.size());
                return NdArray<dtype>(dataPtr, 1, size, PointerPolicy::COPY);
            }
            case 2:
            {
                const auto numRows = static_cast<uint32>(numpyArray.shape(0));
                const auto numCols = static_cast<uint32>(numpyArray.shape(1));
                return NdArray<dtype>(dataPtr, numRows, numCols, PointerPolicy::COPY);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("input array must be no more than 2 dimensional.");
                return {};
            }
        }
    }

    //============================================================================
    /// converts a numpy array to a numcpp NdArray using pybind bindings
    /// Python will still own the underlying data.
    ///
    /// @param numpyArray
    ///
    /// @return NdArray<dtype>
    ///
    template<typename dtype>
    NdArray<dtype> pybind2nc_copy(const pbArray<dtype>& numpyArray)
    {
        const auto dataPtr = numpyArray.data();
        switch (numpyArray.ndim())
        {
            case 0:
            {
                return NdArray<dtype>(dataPtr, 0, 0);
            }
            case 1:
            {
                const auto size = static_cast<uint32>(numpyArray.size());
                return NdArray<dtype>(dataPtr, 1, size);
            }
            case 2:
            {
                const auto numRows = static_cast<uint32>(numpyArray.shape(0));
                const auto numCols = static_cast<uint32>(numpyArray.shape(1));
                return NdArray<dtype>(dataPtr, numRows, numCols);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("input array must be no more than 2 dimensional.");
                return {};
            }
        }
    }

    //============================================================================
    /// converts a numcpp NdArray to numpy array using pybind bindings
    ///
    /// @param inArray: the input array
    ///
    /// @return pybind11::array_t
    ///
    template<typename dtype>
    pbArrayGeneric nc2pybind(const NdArray<dtype>& inArray)
    {
        const Shape                          inShape = inArray.shape();
        const std::vector<pybind11::ssize_t> shape{ static_cast<pybind11::ssize_t>(inShape.rows),
                                                    static_cast<pybind11::ssize_t>(inShape.cols) };
        const std::vector<pybind11::ssize_t> strides{ static_cast<pybind11::ssize_t>(inShape.cols * sizeof(dtype)),
                                                      static_cast<pybind11::ssize_t>(sizeof(dtype)) };
        return pbArrayGeneric(shape, strides, inArray.data());
    }

    //============================================================================
    /// converts a numcpp NdArray to numpy array using pybind bindings
    ///
    /// @param inArray: the input array
    /// @param returnPolicy: the return policy
    ///
    /// @return pybind11::array_t
    ///
    template<typename dtype>
    pbArrayGeneric nc2pybind(NdArray<dtype>& inArray, ReturnPolicy returnPolicy)
    {
        const Shape                          inShape = inArray.shape();
        const std::vector<pybind11::ssize_t> shape{ static_cast<pybind11::ssize_t>(inShape.rows),
                                                    static_cast<pybind11::ssize_t>(inShape.cols) };
        const std::vector<pybind11::ssize_t> strides{ static_cast<pybind11::ssize_t>(inShape.cols * sizeof(dtype)),
                                                      static_cast<pybind11::ssize_t>(sizeof(dtype)) };

        switch (returnPolicy)
        {
            case ReturnPolicy::COPY:
            {
                return nc2pybind(inArray);
            }
            case ReturnPolicy::REFERENCE:
            {
                typename pybind11::capsule reference(inArray.data(), [](void* /*ptr*/) { });
                return pbArrayGeneric(shape, strides, inArray.data(), reference);
            }
            case ReturnPolicy::TAKE_OWNERSHIP:
            {
                typename pybind11::capsule garbageCollect(inArray.dataRelease(),
                                                          [](void* ptr)
                                                          {
                                                              auto* dataPtr = reinterpret_cast<dtype*>(ptr);
                                                              delete[] dataPtr;
                                                          });
                return pbArrayGeneric(shape, strides, inArray.data(), garbageCollect);
            }
            default:
            {
                std::stringstream sstream;
                sstream << "ReturnPolicy " << returnPolicyStringMap.at(returnPolicy) << " has not been implemented yet"
                        << std::endl;
                THROW_INVALID_ARGUMENT_ERROR(sstream.str());
            }
        }
    }
} // namespace nc::pybindInterface
#endif
