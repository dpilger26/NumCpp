/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.3
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
/// Class for dealing with 1D polynomials
///
#pragma once

#include "NumCpp/Core/DtypeInfo.hpp"
#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"
#include "NumCpp/Utils/num2str.hpp"
#include "NumCpp/Utils/power.hpp"

#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nc
{
    namespace polynomial
    {
        //================================================================================
        ///						A one-dimensional polynomial class.
        ///                     A convenience class, used to encapsulate "natural"
        ///                     operations on polynomials
        template<typename dtype>
        class Poly1d
        {
        public:
            //============================================================================
            // Method Description:
            ///						Default Constructor (not very usefull, but needed for other
            ///                     containers.
            ///
            Poly1d() noexcept
            {
                STATIC_ASSERT_ARITHMETIC(dtype);
            }

            //============================================================================
            // Method Description:
            ///						Constructor
            ///
            /// @param      inValues: (polynomial coefficients in ascending order of power if second input is false,
            ///                        polynomial roots if second input is true)
            /// @param      isRoots
            ///
            Poly1d(const NdArray<dtype>& inValues, bool isRoots = false)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                if (inValues.size() > DtypeInfo<uint8>::max())
                {
                    THROW_INVALID_ARGUMENT_ERROR("can only make a polynomial of order " + utils::num2str(DtypeInfo<uint8>::max()));
                }

                if (isRoots)
                {
                    coefficients_.push_back(1);
                    for (auto value : inValues)
                    {
                        NdArray<dtype> coeffs = { -(value), static_cast<dtype>(1) };
                        *this *= Poly1d<dtype>(coeffs, !isRoots);
                    }
                }
                else
                {
                    for (auto value : inValues)
                    {
                        coefficients_.push_back(value);
                    }
                }
            }

            //============================================================================
            // Method Description:
            ///						Returns the area under the curve between the two bounds
            ///
            /// @param a: the lower bound
            /// @param b: the upper bound
            /// @return double
            ///
            double area(double a, double b) const noexcept
            {
                if (a > b)
                {
                    std::swap(a, b);
                }

                auto polyIntegral = integ();
                return polyIntegral(b) - polyIntegral(a);
            }

            //============================================================================
            // Method Description:
            ///						Returns a copy of the polynomial of the new type
            ///
            /// @return Poly1d
            ///
            template<typename dtypeOut>
            Poly1d<dtypeOut> astype() const noexcept
            {
                auto newCoefficients = NdArray<dtypeOut>(1, static_cast<uint32>(coefficients_.size()));

                auto function = [](dtype value) noexcept -> dtypeOut
                {
                    return static_cast<dtypeOut>(value);
                };

                stl_algorithms::transform(coefficients_.begin(), coefficients_.end(), newCoefficients.begin(), function);

                return Poly1d<dtypeOut>(newCoefficients);
            }

            //============================================================================
            // Method Description:
            ///						Returns the Poly1d coefficients
            ///
            /// @return
            ///				NdArray
            ///
            NdArray<dtype> coefficients() const noexcept
            {
                return NdArray<dtype>(coefficients_);
            }

            //============================================================================
            // Method Description:
            ///						Takes the derivative of the polynomial
            ///
            Poly1d<dtype> deriv() const noexcept
            {
                const uint32 numCoefficients = static_cast<uint32>(coefficients_.size());
                if (numCoefficients == 0)
                {
                    return {};
                }
                else if (numCoefficients == 1)
                {
                    return Poly1d<dtype>({ 0 });
                }

                NdArray<dtype> derivativeCofficients(1, numCoefficients - 1);

                uint32 counter = 0;
                for (uint32 i = 1; i < numCoefficients; ++i)
                {
                    derivativeCofficients[counter++] = coefficients_[i] * i;
                }

                return Poly1d<dtype>(derivativeCofficients);
            }

            //============================================================================
            // Method Description:
            ///						Calculates the integral of the polynomial
            ///
            Poly1d<double> integ() const noexcept
            {
                const uint32 numCoefficients = static_cast<uint32>(coefficients_.size());
                if (numCoefficients == 0)
                {
                    return {};
                }

                NdArray<double> integralCofficients(1, numCoefficients + 1);
                integralCofficients[0] = 0.0;

                for (uint32 i = 0; i < numCoefficients; ++i)
                {
                    integralCofficients[i + 1] = static_cast<double>(coefficients_[i]) / static_cast<double>(i + 1);
                }

                return Poly1d<double>(integralCofficients);
            }

            //============================================================================
            // Method Description:
            ///						Returns the order of the Poly1d
            ///
            /// @return
            ///				NdArray
            ///
            uint32 order() const noexcept
            {
                return static_cast<uint32>(coefficients_.size() - 1);
            }

            //============================================================================
            // Method Description:
            ///						Prints the string representation of the Poly1d object
            ///                     to the console
            ///
            void print() const noexcept
            {
                std::cout << *this << std::endl;
            }

            //============================================================================
            // Method Description:
            ///						Converts the polynomial to a string representation
            ///
            /// @return
            ///				Poly1d
            ///
            std::string str() const noexcept
            {
                const uint32 numCoeffients = static_cast<uint32>(coefficients_.size());

                std::string repr = "Poly1d<";
                uint32 power = 0;
                for (auto& coefficient : coefficients_)
                {
                    if (DtypeInfo<dtype>::isInteger())
                    {
                        if (coefficient == 0)
                        {
                            ++power;
                            continue;
                        }
                    }
                    else
                    {
                        if (utils::essentiallyEqual(coefficient, static_cast<dtype>(0.0)))
                        {
                            ++power;
                            continue;
                        }
                    }

                    repr += utils::num2str(coefficient);

                    if (power > 1)
                    {
                        repr += "x^" + utils::num2str(power);
                    }
                    else if (power == 1)
                    {
                        repr += "x";
                    }

                    ++power;

                    if (power < numCoeffients)
                    {
                        repr += " + ";
                    }
                }

                return repr + ">";
            }

            //============================================================================
            // Method Description:
            ///						Evaluates the Poly1D object for the input value
            ///
            /// @param
            ///				inValue
            /// @return
            ///				Poly1d
            ///
            dtype operator()(dtype inValue) const noexcept
            {
                dtype polyValue = 0;
                uint8 power = 0;
                for (auto& coefficient : coefficients_)
                {
                    polyValue += coefficient * utils::power(inValue, power++);
                }

                return polyValue;
            }

            //============================================================================
            // Method Description:
            ///						Adds the two Poly1d objects
            ///
            /// @param
            ///				inOtherPoly
            /// @return
            ///				Poly1d
            ///
            Poly1d<dtype> operator+(const Poly1d<dtype>& inOtherPoly) const noexcept
            {
                return Poly1d<dtype>(*this) += inOtherPoly;
            }

            //============================================================================
            // Method Description:
            ///						Adds the two Poly1d objects
            ///
            /// @param
            ///				inOtherPoly
            /// @return
            ///				Poly1d
            ///
            Poly1d<dtype>& operator+=(const Poly1d<dtype>& inOtherPoly) noexcept
            {
                if (this->coefficients_.size() < inOtherPoly.coefficients_.size())
                {
                    for (size_t i = 0; i < coefficients_.size(); ++i)
                    {
                        coefficients_[i] += inOtherPoly.coefficients_[i];
                    }
                    for (size_t i = coefficients_.size(); i < inOtherPoly.coefficients_.size(); ++i)
                    {
                        coefficients_.push_back(inOtherPoly.coefficients_[i]);
                    }
                }
                else
                {
                    for (size_t i = 0; i < inOtherPoly.coefficients_.size(); ++i)
                    {
                        coefficients_[i] += inOtherPoly.coefficients_[i];
                    }
                }

                return *this;
            }

            //============================================================================
            // Method Description:
            ///						Subtracts the two Poly1d objects
            ///
            /// @param
            ///				inOtherPoly
            /// @return
            ///				Poly1d
            ///
            Poly1d<dtype> operator-(const Poly1d<dtype>& inOtherPoly) const noexcept
            {
                return Poly1d<dtype>(*this) -= inOtherPoly;
            }

            //============================================================================
            // Method Description:
            ///						Subtracts the two Poly1d objects
            ///
            /// @param
            ///				inOtherPoly
            /// @return
            ///				Poly1d
            ///
            Poly1d<dtype>& operator-=(const Poly1d<dtype>& inOtherPoly) noexcept
            {
                if (this->coefficients_.size() < inOtherPoly.coefficients_.size())
                {
                    for (size_t i = 0; i < coefficients_.size(); ++i)
                    {
                        coefficients_[i] -= inOtherPoly.coefficients_[i];
                    }
                    for (size_t i = coefficients_.size(); i < inOtherPoly.coefficients_.size(); ++i)
                    {
                        coefficients_.push_back(-inOtherPoly.coefficients_[i]);
                    }
                }
                else
                {
                    for (size_t i = 0; i < inOtherPoly.coefficients_.size(); ++i)
                    {
                        coefficients_[i] -= inOtherPoly.coefficients_[i];
                    }
                }

                return *this;
            }

            //============================================================================
            // Method Description:
            ///						Multiplies the two Poly1d objects
            ///
            /// @param
            ///				inOtherPoly
            /// @return
            ///				Poly1d
            ///
            Poly1d<dtype> operator*(const Poly1d<dtype>& inOtherPoly) const noexcept
            {
                return Poly1d<dtype>(*this) *= inOtherPoly;
            }

            //============================================================================
            // Method Description:
            ///						Multiplies the two Poly1d objects
            ///
            /// @param
            ///				inOtherPoly
            /// @return
            ///				Poly1d
            ///
            Poly1d<dtype>& operator*=(const Poly1d<dtype>& inOtherPoly) noexcept
            {
                const uint32 finalCoefficientsSize = order() + inOtherPoly.order() + 1;
                std::vector<dtype> coeffsA(finalCoefficientsSize, 0);
                std::vector<dtype> coeffsB(finalCoefficientsSize, 0);
                stl_algorithms::copy(coefficients_.begin(), coefficients_.end(), coeffsA.begin());
                stl_algorithms::copy(inOtherPoly.coefficients_.cbegin(), inOtherPoly.coefficients_.cend(), coeffsB.begin());

                // now multiply out the coefficients
                std::vector<dtype> finalCoefficients(finalCoefficientsSize, 0);
                for (uint32 i = 0; i < finalCoefficientsSize; ++i)
                {
                    for (uint32 k = 0; k <= i; ++k)
                    {
                        finalCoefficients[i] += coeffsA[k] * coeffsB[i - k];
                    }
                }

                this->coefficients_ = finalCoefficients;
                return *this;
            }

            //============================================================================
            // Method Description:
            ///						Raise the Poly1d to an integer power
            ///
            /// @param
            ///				inPower
            /// @return
            ///				Poly1d
            ///
            Poly1d<dtype> operator^(uint32 inPower) const noexcept
            {
                return Poly1d(*this) ^= inPower;
            }

            //============================================================================
            // Method Description:
            ///						Raise the Poly1d to an integer power
            ///
            /// @param
            ///				inPower
            /// @return
            ///				Poly1d
            ///
            Poly1d<dtype>& operator^=(uint32 inPower) noexcept
            {
                if (inPower == 0)
                {
                    coefficients_.clear();
                    coefficients_.push_back(1);
                    return *this;
                }
                else if (inPower == 1)
                {
                    return *this;
                }

                auto thisPoly(*this);
                for (uint32 power = 1; power < inPower; ++power)
                {
                    *this *= thisPoly;
                }

                return *this;
            }

            //============================================================================
            // Method Description:
            ///						io operator for the Poly1d class
            ///
            /// @param      inOStream
            /// @param      inPoly
            /// @return
            ///				std::ostream
            ///
            friend std::ostream& operator<<(std::ostream& inOStream, const Poly1d<dtype>& inPoly) noexcept
            {
                inOStream << inPoly.str() << std::endl;
                return inOStream;
            }

        private:
            std::vector<dtype>      coefficients_{};
        };
    }
}
