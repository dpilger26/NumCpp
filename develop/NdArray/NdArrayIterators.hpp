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
/// Custom iterators for the NdArray class
///
#pragma once

#include "Types.hpp"

#include <iterator>

namespace nc_develop
{
    //================================================================================
    // Class Description:
    ///	Custom const_iterator for NdArray
    template<typename dtype, 
        typename PointerType, 
        typename DifferenceType>
    class NdArrayConstIterator
    {
    private: 
        using self_type = NdArrayConstIterator<dtype, PointerType, DifferenceType>;

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = dtype;
        using pointer = PointerType;
        using reference = const value_type&;
        using difference_type = DifferenceType;

        //============================================================================
        // Method Description:
        ///	Default Constructor
        ///
        NdArrayConstIterator() = default;

        //============================================================================
        // Method Description:
        ///	Constructor
        ///
        /// @param ptr: the iterator pointer
        ///
        explicit NdArrayConstIterator(pointer ptr) noexcept :
            ptr_(ptr)
        {}

        //============================================================================
        // Method Description:
        ///	Iterator dereference
        ///
        /// @return reference
        ///
        reference operator*() const noexcept 
        {
            return *ptr_;
        }

        //============================================================================
        // Method Description:
        ///	Iterator pointer operator
        ///
        /// @return pointer
        ///
        pointer operator->() const noexcept
        {
            return ptr_;
        }

        //============================================================================
        // Method Description:
        ///	Iterator prefix incrament operator
        ///
        /// @return NdArrayConstIterator&
        ///
        self_type& operator++() noexcept 
        {
            ++ptr_;
            return *this;
        }

        //============================================================================
        // Method Description:
        ///	Iterator postfix incrament operator
        ///
        /// @return NdArrayConstIterator
        ///
        self_type operator++(int) noexcept 
        {
            self_type tmp = *this;
            ++*this;
            return tmp;
        }

        //============================================================================
        // Method Description:
        ///	Iterator prefix decrament operator
        ///
        /// @return NdArrayConstIterator&
        ///
        self_type& operator--() noexcept 
        {
            --ptr_;
            return *this;
        }

        //============================================================================
        // Method Description:
        ///	Iterator postfix decrament operator
        ///
        /// @return NdArrayConstIterator
        ///
        self_type operator--(int) noexcept 
        {
            self_type tmp = *this;
            --*this;
            return tmp;
        }

        //============================================================================
        // Method Description:
        ///	Iterator addition assignement operator
        ///
        /// @param offset
        /// @return NdArrayConstIterator&
        ///
        self_type& operator+=(const difference_type offset) noexcept 
        {
            ptr_ += offset;
            return *this;
        }

        //============================================================================
        // Method Description:
        ///	Iterator addition operator
        ///
        /// @param offset
        /// @return NdArrayConstIterator
        ///
        self_type operator+(const difference_type offset) const noexcept 
        {
            self_type tmp = *this;
            return tmp += offset;
        }

        //============================================================================
        // Method Description:
        ///	Iterator subtraction assignement operator
        ///
        /// @param offset
        /// @return NdArrayConstIterator&
        ///
        self_type& operator-=(const difference_type offset) noexcept 
        {
            return *this += -offset;
        }

        //============================================================================
        // Method Description:
        ///	Iterator subtraction operator
        ///
        /// @param offset
        /// @return NdArrayConstIterator
        ///
        self_type operator-(const difference_type offset) const noexcept 
        {
            self_type tmp = *this;
            return tmp -= offset;
        }

        //============================================================================
        // Method Description:
        ///	Iterator difference operator
        ///
        /// @param rhs
        /// @return difference_type
        ///
        difference_type operator-(const self_type& rhs) const noexcept 
        {
            return ptr_ - rhs.ptr_;
        }

        //============================================================================
        // Method Description:
        ///	Iterator access operator
        ///
        /// @param offset
        /// @return reference
        ///
        reference operator[](const difference_type offset) const noexcept 
        {
            return *(*this + offset);
        }

        //============================================================================
        // Method Description:
        ///	Iterator equality operator
        ///
        /// @param rhs
        /// @return bool
        ///
        bool operator==(const self_type& rhs) const noexcept 
        {
            return ptr_ == rhs.ptr_;
        }

        //============================================================================
        // Method Description:
        ///	Iterator not-equality operator
        ///
        /// @param rhs
        /// @return bool
        ///
        bool operator!=(const self_type& rhs) const noexcept 
        {
            return !(*this == rhs);
        }

        //============================================================================
        // Method Description:
        ///	Iterator less than operator
        ///
        /// @param rhs
        /// @return bool
        ///
        bool operator<(const self_type& rhs) const noexcept 
        {
            return ptr_ < rhs.ptr_;
        }

        //============================================================================
        // Method Description:
        ///	Iterator greater than operator
        ///
        /// @param rhs
        /// @return bool
        ///
        bool operator>(const self_type& rhs) const noexcept 
        {
            return rhs < *this;
        }

        //============================================================================
        // Method Description:
        ///	Iterator less than equal operator
        ///
        /// @param rhs
        /// @return bool
        ///
        bool operator<=(const self_type& rhs) const noexcept 
        {
            return !(rhs < *this);
        }

        //============================================================================
        // Method Description:
        ///	Iterator greater than equal operator
        ///
        /// @param rhs
        /// @return bool
        ///
        bool operator>=(const self_type& rhs) const noexcept
        {
            return !(*this < rhs);
        }

    private:
        pointer ptr_{};
    };

    //============================================================================
    // Method Description:
    ///	Iterator addition operator
    ///
    /// @param offset
    /// @param next
    /// @return bool
    ///
    template <class dtype,
        typename PointerType,
        typename DifferenceType>
    NdArrayConstIterator<dtype, PointerType, DifferenceType> operator+(
        typename NdArrayConstIterator<dtype, PointerType, DifferenceType>::difference_type offset,
        NdArrayConstIterator<dtype, PointerType, DifferenceType> next) noexcept
    {
        return next += offset;
    }

    //================================================================================
    // Class Description:
    ///	Custom iterator for NdArray
    template<typename dtype,
        typename PointerType, 
        typename DifferenceType>
    class NdArrayIterator : public NdArrayConstIterator<dtype, PointerType, DifferenceType>
    {
    private:
        using MyBase = NdArrayConstIterator<dtype, PointerType, DifferenceType>;
        using self_type = NdArrayIterator<dtype, PointerType, DifferenceType>;

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = dtype;
        using pointer = PointerType;
        using reference = value_type&;
        using difference_type = DifferenceType;

        using MyBase::MyBase;

        //============================================================================
        // Method Description:
        ///	Iterator dereference
        ///
        /// @return reference
        ///
        reference operator*() const noexcept 
        {
            return const_cast<reference>(MyBase::operator*());
        }

        //============================================================================
        // Method Description:
        ///	Iterator pointer operator
        ///
        /// @return pointer
        ///
        pointer operator->() const noexcept
        {
            return const_cast<pointer>(MyBase::operator->());
        }

        //============================================================================
        // Method Description:
        ///	Iterator prefix incrament operator
        ///
        /// @return NdArrayConstIterator&
        ///
        self_type& operator++() noexcept 
        {
            MyBase::operator++();
            return *this;
        }

        //============================================================================
        // Method Description:
        ///	Iterator postfix incrament operator
        ///
        /// @return NdArrayConstIterator
        ///
        self_type operator++(int) noexcept 
        {
            self_type tmp = *this;
            MyBase::operator++();
            return tmp;
        }

        //============================================================================
        // Method Description:
        ///	Iterator prefix decrament operator
        ///
        /// @return NdArrayConstIterator&
        ///
        self_type& operator--() noexcept 
        {
            MyBase::operator--();
            return *this;
        }

        //============================================================================
        // Method Description:
        ///	Iterator postfix decrament operator
        ///
        /// @return NdArrayConstIterator
        ///
        self_type operator--(int) noexcept 
        {
            self_type tmp = *this;
            MyBase::operator--();
            return tmp;
        }

        //============================================================================
        // Method Description:
        ///	Iterator addition assignement operator
        ///
        /// @param offset
        /// @return NdArrayConstIterator&
        ///
        self_type& operator+=(const difference_type offset) noexcept 
        {
            MyBase::operator+=(offset);
            return *this;
        }

        //============================================================================
        // Method Description:
        ///	Iterator addition operator
        ///
        /// @param offset
        /// @return NdArrayConstIterator
        ///
        self_type operator+(const difference_type offset) const noexcept 
        {
            self_type tmp = *this;
            return tmp += offset;
        }

        //============================================================================
        // Method Description:
        ///	Iterator subtraction assignement operator
        ///
        /// @param offset
        /// @return NdArrayConstIterator&
        ///
        self_type& operator-=(const difference_type offset) noexcept 
        {
            MyBase::operator-=(offset);
            return *this;
        }

        using MyBase::operator-;

        //============================================================================
        // Method Description:
        ///	Iterator difference operator
        ///
        /// @param offset
        /// @return NdArrayConstIterator
        ///
        self_type operator-(const difference_type offset) const noexcept 
        {
            self_type tmp = *this;
            return tmp -= offset;
        }

        //============================================================================
        // Method Description:
        ///	Iterator access operator
        ///
        /// @param offset
        /// @return reference
        ///
        reference operator[](const difference_type offset) const noexcept 
        {
            return const_cast<reference>(MyBase::operator[](offset));
        }
    };

    //============================================================================
    // Method Description:
    ///	Iterator addition operator
    ///
    /// @param offset
    /// @param next
    /// @return NdArrayIterator
    ///
    template <class dtype,
        typename PointerType,
        typename DifferenceType>
    NdArrayIterator<dtype, PointerType, DifferenceType> operator+(
        typename NdArrayIterator<dtype, PointerType, DifferenceType>::difference_type offset,
        NdArrayIterator<dtype, PointerType, DifferenceType> next) noexcept
    {
        return next += offset;
    }
}  // namespace nc
