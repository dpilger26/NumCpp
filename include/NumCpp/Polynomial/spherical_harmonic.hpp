/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
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
/// Special Functions
///
#pragma once

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/NdArray.hpp"

#include "boost/math/special_functions/spherical_harmonic.hpp"

#include <complex>

namespace nc
{
    namespace polynomial
    {
        //============================================================================
        // Method Description:
        ///	Returns the value of the Spherical Harmonic Ynm(theta, phi).
        /// The spherical harmonics Ynm(theta, phi) are the angular portion of the 
        /// solution to Laplace's equation in spherical coordinates where azimuthal
        /// symmetry is not present.
        ///
        /// @param      n: order of the harmonic
        /// @param      m: degree of the harmonic
        /// @param      theta: Azimuthal (longitudinal) coordinate; must be in [0, 2*pi].
        /// @param      phi: Polar (colatitudinal) coordinate; must be in [0, pi].
        /// @return
        ///				double
        ///
        template<typename dtype1, typename dtype2>
        std::complex<double> spherical_harmonic(uint32 n, int32 m,  dtype1 theta, dtype2 phi)
        {
            STATIC_ASSERT_ARITHMETIC(dtype1);
            STATIC_ASSERT_ARITHMETIC(dtype2);

            return boost::math::spherical_harmonic(m, n, static_cast<double>(phi), static_cast<double>(theta));
        }

        //============================================================================
        // Method Description:
        ///	Returns the real part of the Spherical Harmonic Ynm(theta, phi).
        /// The spherical harmonics Ynm(theta, phi) are the angular portion of the 
        /// solution to Laplace's equation in spherical coordinates where azimuthal
        /// symmetry is not present.
        ///
        /// @param      n: order of the harmonic
        /// @param      m: degree of the harmonic
        /// @param      theta: Azimuthal (longitudinal) coordinate; must be in [0, 2*pi].
        /// @param      phi: Polar (colatitudinal) coordinate; must be in [0, pi].
        /// @return
        ///				double
        ///
        template<typename dtype1, typename dtype2>
        double spherical_harmonic_r(uint32 n, int32 m,  dtype1 theta, dtype2 phi)
        {
            STATIC_ASSERT_ARITHMETIC(dtype1);
            STATIC_ASSERT_ARITHMETIC(dtype2);

            return boost::math::spherical_harmonic_r(m, n, static_cast<double>(phi), static_cast<double>(theta));
        }

        //============================================================================
        // Method Description:
        ///	Returns the imaginary part of the Spherical Harmonic Ynm(theta, phi).
        /// The spherical harmonics Ynm(theta, phi) are the angular portion of the 
        /// solution to Laplace's equation in spherical coordinates where azimuthal
        /// symmetry is not present.
        ///
        /// @param      n: order of the harmonic
        /// @param      m: degree of the harmonic
        /// @param      theta: Azimuthal (longitudinal) coordinate; must be in [0, 2*pi].
        /// @param      phi: Polar (colatitudinal) coordinate; must be in [0, pi].
        /// @return
        ///				double
        ///
        template<typename dtype1, typename dtype2>
        double spherical_harmonic_i(uint32 n, int32 m,  dtype1 theta, dtype2 phi)
        {
            STATIC_ASSERT_ARITHMETIC(dtype1);
            STATIC_ASSERT_ARITHMETIC(dtype2);

            return boost::math::spherical_harmonic_i(m, n, static_cast<double>(phi), static_cast<double>(theta));
        }
    } // namespace polynomial
} // namespace nc
