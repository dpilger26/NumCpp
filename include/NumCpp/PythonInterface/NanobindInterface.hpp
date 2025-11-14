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
/// A module for interacting with python with nanobind interface
///
#pragma once

#ifdef NUMCPP_INCLUDE_NANOBIND_PYTHON_INTERFACE

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"

#include <map>
#include <utility>

#include "NumCpp/Core/Enums.hpp"
#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::nanobindInterface
{
    /// Enum for the nanobind array return policy
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
    using nanoArray        = nanobind::ndarray<dtype, nanobind::numpy, nanobind::shape<-1, -1>, nanobind::c_contig>;
    using nanoArrayGeneric = nanobind::ndarray;

    //============================================================================
    /// converts a numpy array to a numcpp NdArray using nanobind bindings
    /// Python will still own the underlying data.
    ///
    /// @param numpyArray
    ///
    /// @return NdArray<dtype>
    ///
    template<typename dtype>
    NdArray<dtype> nanobind2nc(nanoArray<dtype>& numpyArray)
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
    /// converts a numpy array to a numcpp NdArray using nanobind bindings
    /// Python will still own the underlying data.
    ///
    /// @param numpyArray
    ///
    /// @return NdArray<dtype>
    ///
    template<typename dtype>
    NdArray<dtype> nanobind2nc_copy(const nanoArray<dtype>& numpyArray)
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
    /// converts a numcpp NdArray to numpy array using nanobind bindings
    ///
    /// @param inArray: the input array
    ///
    /// @return nanobind::array_t
    ///
    template<typename dtype>
    nanoArrayGeneric nc2nanobind(const NdArray<dtype>& inArray)
    {
        const Shape                          inShape = inArray.shape();
        const std::vector<nanobind::ssize_t> shape{ static_cast<nanobind::ssize_t>(inShape.rows),
                                                    static_cast<nanobind::ssize_t>(inShape.cols) };
        const std::vector<nanobind::ssize_t> strides{ static_cast<nanobind::ssize_t>(inShape.cols * sizeof(dtype)),
                                                      static_cast<nanobind::ssize_t>(sizeof(dtype)) };
        return nanoArrayGeneric(shape, strides, inArray.data());
    }

    //============================================================================
    /// converts a numcpp NdArray to numpy array using nanobind bindings
    ///
    /// @param inArray: the input array
    /// @param returnPolicy: the return policy
    ///
    /// @return nanobind::array_t
    ///
    template<typename dtype>
    nanoArrayGeneric nc2nanobind(NdArray<dtype>& inArray, ReturnPolicy returnPolicy)
    {
        const Shape                          inShape = inArray.shape();
        const std::vector<nanobind::ssize_t> shape{ static_cast<nanobind::ssize_t>(inShape.rows),
                                                    static_cast<nanobind::ssize_t>(inShape.cols) };
        const std::vector<nanobind::ssize_t> strides{ static_cast<nanobind::ssize_t>(inShape.cols * sizeof(dtype)),
                                                      static_cast<nanobind::ssize_t>(sizeof(dtype)) };

        switch (returnPolicy)
        {
            case ReturnPolicy::COPY:
            {
                return nc2nanobind(inArray);
            }
            case ReturnPolicy::REFERENCE:
            {
                typename nanobind::capsule reference(inArray.data(), [](void* /*ptr*/) { });
                return nanoArrayGeneric(shape, strides, inArray.data(), reference);
            }
            case ReturnPolicy::TAKE_OWNERSHIP:
            {
                typename nanobind::capsule garbageCollect(inArray.dataRelease(),
                                                          [](void* ptr)
                                                          {
                                                              auto* dataPtr = reinterpret_cast<dtype*>(ptr);
                                                              delete[] dataPtr;
                                                          });
                return nanoArrayGeneric(shape, strides, inArray.data(), garbageCollect);
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
} // namespace nc::nanobindInterface
#endif
