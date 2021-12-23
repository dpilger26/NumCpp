/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2021 David Pilger
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
/// Functions for working with NdArrays
///
#pragma once

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Internal/Error.hpp"

#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <vector>

namespace nc
{
    //============================================================================
    // Method Description:
    ///	Return an array drawn from elements in choiceVec, depending on conditions.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.select.html?highlight=select#numpy.select
    ///
    /// @param condVec The vector of conditions which determine from which array in choiceVec
    ///                the output elements are taken. When multiple conditions are satisfied, 
    ///                the first one encountered in choiceVec is used.
    /// @param choiceVec The vector of array pointers from which the output elements are taken. 
    ///                  It has to be of the same length as condVec.
    /// @param defaultValue The element inserted in output when all conditions evaluate to False
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> select(const std::vector<const NdArray<bool>*>& condVec, 
        const std::vector<const NdArray<dtype>*>& choiceVec, dtype defaultValue = dtype{0})
    {
        if (choiceVec.size() != condVec.size())
        {
            THROW_INVALID_ARGUMENT_ERROR("condVec and choiceVec need to be the same size");
        }

        if (choiceVec.size() == 0)
        {
            THROW_INVALID_ARGUMENT_ERROR("choiceVec is size 0");
        }

        auto theShape = condVec.front()->shape();
        for (const auto cond : condVec)
        {
            const auto& theCond = *cond;
            if (theCond.shape() != theShape)
            {
                THROW_INVALID_ARGUMENT_ERROR("all NdArrays of the condVec must be the same shape");
            }
        }

        for (const auto choice : choiceVec)
        {
            const auto& theChoice = *choice;
            if (theChoice.shape() != theShape)
            {
                THROW_INVALID_ARGUMENT_ERROR("all NdArrays of the choiceVec must be the same shape, and the same as condVec");
            }
        }

        using size_type = typename NdArray<dtype>::size_type;
        constexpr auto nullChoice = std::numeric_limits<size_type>::max();

        NdArray<size_type> choiceIndices(theShape);
        choiceIndices.fill(nullChoice);
        for (size_type condIdx = 0; condIdx < condVec.size(); ++condIdx)
        {
            const auto& theCond = *condVec[condIdx];
            for (size_type i = 0; i < theCond.size(); ++i)
            {
                if (theCond[i] && choiceIndices[i] == nullChoice)
                {
                    choiceIndices[i] = condIdx;
                }
            }
        }

        NdArray<dtype> result(theShape);
        result.fill(defaultValue);
        for (size_type i = 0; i < choiceIndices.size(); ++i)
        {
            const auto choiceIndex = choiceIndices[i];
            if (choiceIndex != nullChoice)
            {
                const auto& theChoice = *choiceVec[choiceIndex];
                result[i] = theChoice[i];
            }
        }

        return result;
    }

    //============================================================================
    // Method Description:
    ///	Return an array drawn from elements in choiceList, depending on conditions.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.select.html?highlight=select#numpy.select
    ///
    /// @param condList The list of conditions which determine from which array in choiceList
    ///                the output elements are taken. When multiple conditions are satisfied, 
    ///                the first one encountered in choiceList is used.
    /// @param choiceList The list of array pointers from which the output elements are taken. 
    ///                  It has to be of the same length as condVec.
    /// @param defaultValue The element inserted in output when all conditions evaluate to False
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> select(const std::vector<NdArray<bool>>& condList, 
        const std::vector<NdArray<dtype>>& choiceList, 
        dtype defaultValue = dtype{0})
    {
        std::vector<const NdArray<bool>*> condVec;
        condVec.reserve(condList.size());
        for (auto& cond : condList)
        {
            condVec.push_back(&cond);
        }

        std::vector<const NdArray<dtype>*> choiceVec;
        choiceVec.reserve(choiceList.size());
        for (auto& choice : choiceList)
        {
            choiceVec.push_back(&choice);
        }

        return select(condVec, choiceVec, defaultValue);
    }
} // namespace nc
