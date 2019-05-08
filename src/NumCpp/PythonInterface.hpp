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
/// A module for interacting with python
///
#pragma once

#ifdef INCLUDE_BOOST_PYTHON_INTERFACE
#include"NumCpp/BoostNumpyNdarrayHelper.hpp"
#endif
#include"NumCpp/NdArray.hpp"
#include"NumCpp/Shape.hpp"

#include<iostream>
#include<map>
#include<stdexcept>
#include<utility>

#ifdef INCLUDE_BOOST_PYTHON_INTERFACE
#include"boost/python.hpp"
#include"boost/python/numpy.hpp"
#endif
#ifdef INCLUDE_PYBIND_PYTHON_INTERFACE
#include"pybind11/pybind11.h"
#include"pybind11/numpy.h"
#endif

namespace nc
{
#ifdef INCLUDE_BOOST_PYTHON_INTERFACE
    //============================================================================
    ///						Converts from a boost ndarray to a NumCpp NdArray<T>
    ///
    /// @param      inArray
    ///
    /// @return     NdArray<T>
    ///
    template<typename dtype>
    NdArray<dtype> boost2Nc(const boost::python::numpy::ndarray& inArray)
    {
        BoostNdarrayHelper helper(inArray);
        if (helper.numDimensions() > 2)
        {
            std::string errStr = "ERROR: Can only convert 1 and 2 dimensional arrays.";
            std::cerr << errStr << std::endl;
            throw std::runtime_error(errStr);
        }

        Shape arrayShape;
        if (helper.numDimensions() == 1)
        {
            arrayShape.rows = 1;
            arrayShape.cols = static_cast<uint32>(helper.shape().front());

            NdArray<dtype> returnArray(arrayShape);
            for (uint32 i = 0; i < arrayShape.size(); ++i)
            {
                returnArray[i] = static_cast<dtype>(helper(i));
            }

            return returnArray;
        }
        else
        {
            arrayShape.rows = static_cast<uint32>(helper.shape().front());
            arrayShape.cols = static_cast<uint32>(helper.shape()[1]);

            NdArray<dtype> returnArray(arrayShape);
            uint32 i = 0;
            for (uint32 row = 0; row < arrayShape.rows; ++row)
            {
                for (uint32 col = 0; col < arrayShape.cols; ++col)
                {
                    returnArray[i++] = static_cast<dtype>(helper(row, col));
                }
            }

            return returnArray;
        }
    }

    //============================================================================
    ///						Converts from a NumCpp NdArray<T> to a boost ndarray
    ///
    /// @param      inArray
    ///
    /// @return     ndarray
    ///
    template<typename dtype>
    boost::python::numpy::ndarray nc2Boost(const NdArray<dtype>& inArray)
    {
        const Shape inShape = inArray.shape();
        boost::python::tuple shape = boost::python::make_tuple(inShape.rows, inShape.cols);
        BoostNdarrayHelper newNdArrayHelper(shape);

        for (uint32 row = 0; row < inShape.rows; ++row)
        {
            for (uint32 col = 0; col < inShape.cols; ++col)
            {
                newNdArrayHelper(row, col) = static_cast<double>(inArray(row, col));
            }
        }
        return newNdArrayHelper.getArray();
    }

    //============================================================================
    ///						converts a boost python list to a std::vector
    ///
    /// @param      inList
    ///
    /// @return     std::vector<T>
    ///
    template<typename T>
    std::vector<T> list2vector(const boost::python::list& inList)
    {
        return std::vector<T>(boost::python::stl_input_iterator<T>(inList), boost::python::stl_input_iterator<T>());
    }

    //============================================================================
    ///						converts a std::vector to a boost python list
    ///
    /// @param      inVector
    ///
    /// @return     boost::python::list
    ///
    template <typename T>
    boost::python::list vector2list(std::vector<T>& inVector)
    {
        boost::python::list outList;
        for (auto& value : inVector)
        {
            outList.append(value);
        }

        return outList;
    }

    //============================================================================
    ///						converts a std::map in to a boost python dictionary
    ///
    /// @param      inMap
    ///
    /// @return     boost::python::dict
    ///
    template <class Key, class Value>
    boost::python::dict map2dict(const std::map<Key, Value>& inMap)
    {
        boost::python::dict dictionary;
        for (auto& keyValue : inMap)
        {
            dictionary[keyValue.first] = keyValue.second;
        }
        return dictionary;
    }
#endif

#ifdef INCLUDE_PYBIND_PYTHON_INTERFACE
    //============================================================================
    ///						converts a numpy array to a numcpp NdArray using pybind bindings
    ///                     Python will still own the underlying data.
    ///
    /// @param      numpyArray
    ///
    /// @return     NdArray<dtype>
    ///
    template<typename dtype>
    NdArray<dtype> pybind2nc(pybind11::array_t<dtype, pybind11::array::c_style>& numpyArray)
    {
        switch (numpyArray.ndim())
        {
            dtype* dataPtr = numpyArray.mutable_data();

            case 0:
            {
                return NdArray<dtype>(dataPtr, 0, 0, false);
            }
            case 1:
            {
                uint32 size = static_cast<uint32>(numpyArray.size());
                return NdArray<dtype>(dataPtr, 1, size, false);
            }
            case 2:
            {
                uint32 numRows = static_cast<uint32>(numpyArray.shape[0]);
                uint32 numCols = static_cast<uint32>(numpyArray.shape[1]);
                return NdArray<dtype>(dataPtr, numRows, numCols, false);
            }
            default:
            {
                throw std::invalid_argument("ERROR: input array must be no more than 2 dimensional.");
            }
        }
    }

    //============================================================================
    ///						converts a numcpp NdArray to numpy array using pybind bindings
    ///
    /// @param     inArray: the input array
    /// @param     transferOwnership: whether or not to transfer ownership to python. 
    ///                               Requires that the NdArray owns its data.
    ///
    /// @return    pybind11::array_t
    ///
    template<typename dtype>
    pybind11::array_t<dtype> nc2pybind(NdArray<dtype>& inArray, bool transferOwnership = true)
    {
        Shape inShape = inArray.shape();
        std::vector<pybind11::ssize_t> shape{ inShape.rows, inShape.cols };
        std::vector<pybind11::ssize_t> strides{ inShape.cols * sizeof(dtype), sizeof(dtype) };

        if (inArray.ownsInternalData() && transferOwnership)
        {
            typename py::capsule transfer(inArray.begin(), [](void* ptr) { delete[] ptr; });  // python now owns the memory
            return pybind11::array_t<dtype>(shape, strides, inArray.dataRelease(), transfer);
        }
        else
        {
            typename py::capsule reference(inArray.begin(), [](void* ptr) {});  // original owner still owns the memory, passing back reference
            return pybind11::array_t<dtype>(shape, strides, inArray.data(), reference);
        }
    }
#endif
}
