/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2022 David Pilger
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
/// Hamming EDAC encoding https://en.wikipedia.org/wiki/Hamming_code
///

#pragma once

#ifndef NUMCPP_NO_USE_BOOST

#include <bitset>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "boost/dynamic_bitset.hpp"

#include "NumCpp/Core/Internal/TypeTraits.hpp"

namespace nc
{
    namespace edac::detail
    {
        //============================================================================
        // Method Description:
        /// @brief Tests if value is a power of two
        ///
        /// @param n integer value
        /// @return bool true if value is a power of two, else false
        ///
        template<
            typename IntType,
            std::enable_if_t<std::is_integral_v<IntType>, int> = 0
        >
        constexpr bool isPowerOfTwo(IntType n) noexcept
        {
            // Returns true if the given non-negative integer n is a power of two.
            return n != 0 && (n & (n - 1)) == 0;
        }

        //============================================================================
        // Method Description:
        /// Calculates the next power of two after n
        /// >>> _next_power_of_two(768)
        ///     1024
        /// >>> _next_power_of_two(4)
        ///     8
        ///
        /// @param n integer value
        /// @return next power of two
        /// @exception std::invalid_argument if input value is less than zero
        ////
        template<
            typename IntType,
            std::enable_if_t<std::is_integral_v<IntType>, int> = 0
        >
        std::size_t nextPowerOfTwo(IntType n)
        {
            if (n < 0)
            {
                throw std::invalid_argument("Input value must be greater than or equal to zero.");
            }

            if (isPowerOfTwo(n))
            {
                return static_cast<std::size_t>(n) << 1;
            }

            return static_cast<std::size_t>(std::pow(2, std::ceil(std::log2(n))));
        }

        //============================================================================
        // Method Description:
        /// Calculates the first n powers of two
        ///
        /// @param n integer value
        /// @return first n powers of two
        /// @exception std::bad_alloc if unable to allocate for return vector
        ///
        template<
            typename IntType,
            std::enable_if_t<std::is_integral_v<IntType>, int> = 0
        >
        std::vector<std::size_t> powersOfTwo(IntType n)
        {
            auto i = std::size_t{ 0 };
            auto power = std::size_t{ 1 };
            auto powers = std::vector<std::size_t>();
            powers.reserve(n);

            while (i < static_cast<std::size_t>(n))
            {
                powers.push_back(power);
                power <<= 1;
                ++i;
            }

            return powers;
        }

        //============================================================================
        // Method Description:
        /// Calculates the number of needed Hamming SECDED parity bits to encode the data
        ///
        /// @param numDataBits the number of data bits to encode
        /// @return number of Hamming SECDED parity bits
        /// @exception std::invalid_argument if input value is less than zero
        /// @exception std::runtime_error if the number of data bits does not represent a valid Hamming SECDED code
        ///
        template<
            typename IntType,
            std::enable_if_t<std::is_integral_v<IntType>, int> = 0
        >
        std::size_t numSecdedParityBitsNeeded(IntType numDataBits)
        {
            const auto n = nextPowerOfTwo(numDataBits);
            const auto lowerBin = static_cast<std::size_t>(std::floor(std::log2(n)));
            const auto upperBin = lowerBin + 1;
            const auto dataBitBoundary = n - lowerBin - 1;
            const auto numParityBits = numDataBits <= dataBitBoundary ? lowerBin + 1 : upperBin + 1;

            if (!isPowerOfTwo(numParityBits + numDataBits))
            {
                throw std::runtime_error("input number of data bits is not a valid Hamming SECDED code configuration.");
            }

            return numParityBits;
        }

        //============================================================================
        // Method Description:
        /// Returns the indices of all data bits covered by a specified parity bit in a bitstring
        /// of length numDataBits. The indices are relative to DATA BITSTRING ITSELF, NOT including
        /// parity bits.
        ///
        /// @param numDataBits the number of data bits to encode
        /// @param parityBit the parity bit number
        /// @return number of Hamming SECDED parity bits
        /// @exception std::invalid_argument if parityBit is not a power of two
        /// @exception std::bad_alloc if unable to allocate return vector
        ///
        template<
            typename IntType1,
            typename IntType2,
            std::enable_if_t<std::is_integral_v<IntType1>, int> = 0,
            std::enable_if_t<std::is_integral_v<IntType2>, int> = 0
        >
        std::vector<std::size_t> dataBitsCovered(IntType1 numDataBits, IntType2 parityBit)
        {
            if (!isPowerOfTwo(parityBit))
            {
                throw std::invalid_argument("All hamming parity bits are indexed by powers of two.");
            }

            std::size_t dataIndex = 1; // bit we're currently at in the DATA bitstring
            std::size_t totalIndex = 3; // bit we're currently at in the OVERALL bitstring
            auto parityBit_ = static_cast<std::size_t>(parityBit);

            auto indices = std::vector<std::size_t>();
            indices.reserve(numDataBits); // worst case

            while (dataIndex <= static_cast<std::size_t>(numDataBits))
            {
                const auto currentBitIsData = !isPowerOfTwo(totalIndex);
                if (currentBitIsData && (totalIndex % (parityBit_ << 1)) >= parityBit_)
                {
                    indices.push_back(dataIndex - 1); // adjust output to be zero indexed
                }

                dataIndex += currentBitIsData ? 1 : 0;
                ++totalIndex;
            }

            return indices;
        }

        //============================================================================
        // Method Description:
        /// Calculates the overall parity of the data, assumes last bit is the parity bit itself
        ///
        /// @param data the data word
        /// @return overall parity bit value
        ///
        template<std::size_t DataBits>
        constexpr bool calculateParity(const std::bitset<DataBits>& data) noexcept
        {
            bool parity = false;
            for (std::size_t i = 0; i < DataBits - 1; ++i)
            {
                parity ^= data[i];
            }

            return parity;
        }

        //============================================================================
        // Method Description:
        /// Calculates the overall parity of the data, assumes last bit is the parity bit itself
        ///
        /// @param data the data word
        /// @return overall parity bit value
        ///
        inline bool calculateParity(const boost::dynamic_bitset<>& data) noexcept
        {
            bool parity = false;
            for (std::size_t i = 0; i < data.size() - 1; ++i)
            {
                parity ^= data[i];
            }

            return parity;
        }

        //============================================================================
        // Method Description:
        /// Calculates the specified Hamming parity bit (1, 2, 4, 8, etc.) for the given data.
        /// Assumes even parity to allow for easier computation of parity using XOR.
        ///
        /// @param data the data word
        /// @param parityBit the parity bit number
        /// @return parity bit value
        /// @exception std::invalid_argument if parityBit is not a power of two
        /// @exception std::bad_alloc if unable to allocate return vector
        ///
        template<
            std::size_t DataBits,
            typename IntType,
            std::enable_if_t<std::is_integral_v<IntType>, int> = 0
        >
        bool calculateParity(const std::bitset<DataBits>& data, IntType parityBit)
        {
            bool parity = false;
            for (const auto i : dataBitsCovered(DataBits, parityBit))
            {
                parity ^= data[i];
            }

            return parity;
        }

        //============================================================================
        // Method Description:
        /// Checks that the number of DataBits and EncodedBits are consistent
        ///
        /// @return the number of parity bits
        /// @exception std::runtime_error if DataBits and EncodedBits are not consistent
        /// @exception std::runtime_error if the number of data bits does not represent a valid Hamming SECDED code
        ///
        template<
            std::size_t DataBits,
            std::size_t EncodedBits,
            std::enable_if_t<greaterThan_v<EncodedBits, DataBits>, int> = 0
        >
        std::size_t checkBitsConsistent()
        {
            const auto numParityBits = detail::numSecdedParityBitsNeeded(DataBits);
            if (numParityBits + DataBits != EncodedBits)
            {
                throw std::runtime_error("DataBits and EncodedBits are not consistent");
            }

            return numParityBits;
        }

        //============================================================================
        // Method Description:
        /// Returns the Hamming SECDED decoded bits from the endoded bits. Assumes that the 
        ///                DataBits and EncodedBits have been checks for consistancy already
        ///
        /// @param encodedBits the Hamming SECDED encoded word
        /// @return data bits from the encoded word
        ///
        template<
            std::size_t DataBits,
            std::size_t EncodedBits,
            std::enable_if_t<greaterThan_v<EncodedBits, DataBits>, int> = 0
        >
        std::bitset<DataBits> extractData(const std::bitset<EncodedBits>& encodedBits) noexcept
        {
            auto dataBits = std::bitset<DataBits>();

            std::size_t dataIndex = 0;
            for (std::size_t encodedIndex = 0; encodedIndex < EncodedBits; ++encodedIndex)
            {
                if (!isPowerOfTwo(encodedIndex + 1))
                {
                    dataBits[dataIndex++] = encodedBits[encodedIndex];
                    if (dataIndex == DataBits)
                    {
                        break;
                    }
                }
            }

            return dataBits;
        }
    } // namespace edac::detail

    namespace edac
    {
        //============================================================================
        // Method Description:
        /// Returns the Hamming SECDED encoded bits for the data bits
        ///
        /// @param dataBits the data bits to encode
        /// @return encoded data bits
        /// @exception std::runtime_error if the number of data bits does not represent a valid Hamming SECDED code
        ///
        template<std::size_t DataBits>
        boost::dynamic_bitset<> encode(const std::bitset<DataBits>& dataBits)
        {
            const auto numParityBits = detail::numSecdedParityBitsNeeded(DataBits);
            const auto numEncodedBits = numParityBits + DataBits;

            auto encodedBits = boost::dynamic_bitset<>(numEncodedBits);

            // set the parity bits
            for (const auto parityBit : detail::powersOfTwo(numParityBits - 1)) // -1 because overall parity will be calculated seperately later
            {
                encodedBits[parityBit - 1] = detail::calculateParity(dataBits, parityBit);
            }

            // set the data bits, switch to 1 based to make things easier for isPowerOfTwo
            std::size_t dataBitIndex = 0;
            for (std::size_t bitIndex = 1; bitIndex <= numEncodedBits - 1; ++bitIndex) // -1 to account for the overall parity bit
            {
                if (!detail::isPowerOfTwo(bitIndex))
                {
                    encodedBits[bitIndex - 1] = dataBits[dataBitIndex++];
                }
            }

            // compute and set overall parity for the entire encoded data (not including the overall parity bit itself)
            encodedBits[numEncodedBits - 1] = detail::calculateParity(encodedBits); // overall parity at the end

            // all done!
            return encodedBits;
        }

        //============================================================================
        // Method Description:
        ///	Returns the Hamming SECDED decoded bits for the enocoded bits
        /// https://en.wikipedia.org/wiki/Hamming_code
        ///
        /// @param encodedBits the encoded bits to decode
        /// @param decodedBits the output decoded bits
        /// @return int status (0=no errors, 1=1 corrected error, 2=2 errors detected)
        /// @exception std::runtime_error if DataBits and EncodedBits are not consistent
        /// @exception std::runtime_error if the number of data bits does not represent a valid Hamming SECDED code
        ///
        template<
            std::size_t DataBits,
            std::size_t EncodedBits,
            std::enable_if_t<greaterThan_v<EncodedBits, DataBits>, int> = 0
        >
        int decode(std::bitset<EncodedBits> encodedBits, std::bitset<DataBits>& decodedBits)
        {
            const auto numParityBits = detail::checkBitsConsistent<DataBits, EncodedBits>();

            // the data bits, which may be corrupted
            decodedBits = detail::extractData<DataBits>(encodedBits);

            // check the overall parity bit
            const auto overallExpected = detail::calculateParity(encodedBits);
            const auto overallActual = encodedBits[EncodedBits - 1];
            const auto overallCorrect = overallExpected == overallActual;

            // check individual parities - each parity bit's index (besides overall parity) is a power of two
            std::size_t indexOfError = 0;
            for (const auto parityBit : detail::powersOfTwo(numParityBits - 1))
            {
                const auto expected = detail::calculateParity(decodedBits, parityBit);
                const auto actual = encodedBits[parityBit - 1]; // -1 because parityBit is 1 based
                if (expected != actual)
                {
                    indexOfError += parityBit;
                }
            }

            // attempt to repair a single flipped bit or throw exception if more than one
            if (overallCorrect && indexOfError != 0)
            {
                // two errors found
                return 2;
            }
            else if (!overallCorrect && indexOfError != 0)
            {
                // one error found, flip the bit in error and we're good
                encodedBits.flip(indexOfError - 1);
                decodedBits = detail::extractData<DataBits>(encodedBits);
                return 1;
            }

            return 0;
        }
    } // namespace edac
} // namespace nc
#endif // #ifndef NUMCPP_NO_USE_BOOST
