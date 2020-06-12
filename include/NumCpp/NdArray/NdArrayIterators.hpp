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
/// Custom iterators for the NdArray class
///
#pragma once

#include "NumCpp/Core/Types.hpp"

#include <iterator>

namespace nc
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
        NdArrayConstIterator(pointer ptr) noexcept :
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
        const pointer operator->() const noexcept
        {
            return const_cast<const pointer>(MyBase::operator->());
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
        /// @param rhs
        /// @return difference_type
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
    /// @return bool
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

    //================================================================================
    // Class Description:
    ///	Custom column const_iterator for NdArray
    template<typename dtype, 
        typename SizeType,
        typename PointerType, 
        typename DifferenceType>
    class NdArrayConstColumnIterator
    {
    private: 
        using self_type = NdArrayConstColumnIterator<dtype, SizeType, PointerType, DifferenceType>;

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = dtype;
        using size_type         = SizeType;
        using pointer           = PointerType;
        using reference         = const value_type&;
        using difference_type   = DifferenceType;

        //============================================================================
        // Method Description:
        ///	Default Constructor
        ///
        NdArrayConstColumnIterator() = default;

        //============================================================================
        // Method Description:
        ///	Constructor
        ///
        /// @param ptr: the iterator pointer
        ///
        NdArrayConstColumnIterator(pointer ptr, SizeType numRows, SizeType numCols) noexcept :
            ptr_(ptr),
            currPtr_(ptr),
            numRows_(static_cast<difference_type>(numRows)),
            numCols_(static_cast<difference_type>(numCols))
        {}

        //============================================================================
        // Method Description:
        ///	Iterator dereference
        ///
        /// @return reference
        ///
        reference operator*() const noexcept
        {
            return *currPtr_;
        }

        //============================================================================
        // Method Description:
        ///	Iterator pointer operator
        ///
        /// @return pointer
        ///
        pointer operator->() const noexcept
        {
            return currPtr_;
        }

        //============================================================================
        // Method Description:
        ///	Iterator prefix incrament operator
        ///
        /// @return NdArrayConstIterator&
        ///
        self_type& operator++() noexcept
        {
            return *this += 1;
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
            return *this -= 1;
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
            auto currLocation = pointerToRowCol(currPtr_);
            if (currLocation.row >= numRows_)
            {
                const auto idx = rowOrderPtrToColOrderIdx(currPtr_);
                std::cout << std::dec << "idx = " << idx << '\n';
                currLocation = offsetToLocationDiff(idx);
            }
            std::cout << std::dec << "currLocation = [" << currLocation.row << ", " << currLocation.col << "]\n";
            const auto diffLocation = offsetToLocationDiff(offset);
            std::cout << std::dec << "diffLocation = [" << diffLocation.row << ", " << diffLocation.col << "]\n";

            auto newLocation = currLocation + diffLocation;
            std::cout << std::dec << "newLocation pre wrap = [" << newLocation.row << ", " << newLocation.col << "]\n";

            if (newLocation.row < 0)
            {
                newLocation.row += numRows_;
                newLocation.col -= 1;
            }
            else if (newLocation.row >= numRows_)
            {
                newLocation.row -= numRows_;
                newLocation.col += 1;
            }

            std::cout << std::dec << "newLocation post wrap = [" << newLocation.row << ", " << newLocation.col << "]\n";

            currPtr_ = rowColToPointer(newLocation);
            std::cout << std::hex << "ptr = " << ptr_ << '\n';
            std::cout << std::hex << "currPtr = " << currPtr_ << "\n\n";
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
            return rowOrderPtrToColOrderIdx(rhs.currPtr_) - rowOrderPtrToColOrderIdx(currPtr_);
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
            return currPtr_ == currPtr_;
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
            return *this - rhs < 0;
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
            return *this - rhs > 0;
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
        pointer         ptr_{};
        pointer         currPtr_{};
        difference_type numRows_{ 0 };
        difference_type numCols_{ 0 };

        struct Location
        {
            difference_type row;
            difference_type col;

            Location& operator+=(const Location& rhs) noexcept
            {
                row += rhs.row;
                col += rhs.col;
                return *this;
            }

            Location operator+(const Location& rhs) const noexcept
            {
                return Location(*this) += rhs;
            }

            Location& operator-=(const Location& rhs) noexcept
            {
                row -= rhs.row;
                col -= rhs.col;
                return *this;
            }

            Location operator-(const Location& rhs) const noexcept
            {
                return Location(*this) -= rhs;
            }
        };

        //============================================================================
        // Method Description:
        ///	Converts a pointer to a row and column location in the 2d array
        ///
        /// @param ptr
        /// @return Location
        ///
        Location pointerToRowCol(pointer ptr) const noexcept
        {
            const auto idx = ptr - ptr_;
            const auto row = idx / numCols_;
            const auto col = idx % numCols_;
            return { row, col };
        }

        //============================================================================
        // Method Description:
        ///	Converts an offset to a row and column location difference column based
        ///
        /// @param offset
        /// @return Location
        ///
        Location offsetToLocationDiff(difference_type offset) const noexcept
        {
            const auto row = offset % numRows_;
            const auto col = offset / numRows_;
            return { row, col };
        }

        //============================================================================
        // Method Description:
        ///	Converts a pointer to a row and column location in the 2d array
        ///
        /// @param row
        /// @param col
        /// @return pointer
        ///
        pointer rowColToPointer(Location location) const noexcept
        {
            const auto idx = location.row + location.col * numRows_;
            return ptr_ + idx;
        }

        //============================================================================
        // Method Description:
        ///	Converts a pointer to a column order idx
        ///
        /// @param ptr
        /// @return difference_type
        ///
        difference_type rowOrderPtrToColOrderIdx(pointer ptr) const noexcept
        {
            const auto location = pointerToRowCol(ptr);
            return location.col * numRows_ + location.row;
        }
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
        typename SizeType,
        typename PointerType,
        typename DifferenceType>
    NdArrayConstColumnIterator<dtype, SizeType, PointerType, DifferenceType> operator+(
        typename NdArrayConstColumnIterator<dtype, SizeType, PointerType, DifferenceType>::difference_type offset,
        NdArrayConstColumnIterator<dtype, SizeType, PointerType, DifferenceType> next) 
    {
        return next += offset;
    }

    //================================================================================
    // Class Description:
    ///	Custom column iterator for NdArray
    template<typename dtype,
        typename SizeType,
        typename PointerType, 
        typename DifferenceType>
    class NdArrayColumnIterator : public NdArrayConstColumnIterator<dtype, SizeType, PointerType, DifferenceType>
    {
    private:
        using MyBase    = NdArrayConstColumnIterator<dtype, SizeType, PointerType, DifferenceType>;
        using self_type = NdArrayColumnIterator<dtype, SizeType, PointerType, DifferenceType>;

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = dtype;
        using size_type         = SizeType;
        using pointer           = PointerType;
        using reference         = value_type&;
        using difference_type   = DifferenceType;

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
        const pointer operator->() const noexcept
        {
            return const_cast<const pointer>(MyBase::operator->());
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
        /// @param rhs
        /// @return difference_type
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
    /// @return bool
    ///
    template <class dtype,
        typename SizeType,
        typename PointerType,
        typename DifferenceType>
        NdArrayColumnIterator<dtype, SizeType, PointerType, DifferenceType> operator+(
        typename NdArrayColumnIterator<dtype, SizeType, PointerType, DifferenceType>::difference_type offset,
            NdArrayColumnIterator<dtype, SizeType, PointerType, DifferenceType> next) noexcept
    {
        return next += offset;
    }
}
