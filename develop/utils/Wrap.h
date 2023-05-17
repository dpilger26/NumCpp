// UNCLASSIFIED
#pragma once

#include <cmath>

#include "boost/math/constants/constants.hpp"

namespace chimera::algs::common::utils
{
	/**
	 * @brief Wrap the input angle to [-pi, pi]
	 *
	 * @params: angle: in radians
	 * @returns Wrapped angle
	 */
	[[nodiscard]] inline double Wrap(double angle) noexcept
	{
		angle = std::fmod(angle + boost::math::constants::pi<double>(), boost::math::constants::two_pi<double>());
		if(angle < 0.)
		{
			angle += boost::math::constants::two_pi<double>();
		}

		return angle - boost::math::constants::pi<double>();
	}

	/**
	 * @brief Wrap the input angle to [0, 2*pi]
	 *
	 * @params: angle: in radians
	 * @returns Wrapped angle
	 */
	[[nodiscard]] inline double Wrap2Pi(double angle) noexcept
	{
		angle = std::fmod(angle, boost::math::constants::two_pi<double>());
		if(angle < 0.)
		{
			angle += boost::math::constants::two_pi<double>();
		}

		return angle;
	}
} // namespace chimera::algs::common::utils

// UNCLASSIFIED
