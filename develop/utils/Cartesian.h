// UNCLASSIFIED
#pragma once

#include <cmath>
#include <iostream>

#include "algs/common/utils/Sqr.h"

namespace chimera::algs::common::utils
{
	/**
	 * @brief Cartensian coordinates
	 */
	class Cartesian
	{
	public:
		/**
		 * @brief Default Constructor
		 */
		Cartesian() noexcept = default;

		/**
		 * @brief Constructor
		 *
		 * @param: x: the x component
		 * @param: y: the y component
		 * @param: z: the z component
		 */
		constexpr Cartesian(double x, double y, double z = 0.0) noexcept : x_(x), y_(y), z_(z)
		{
		}

		/**
		 * @brief Copy Constructor
		 *
		 * @param: other: the other Cartesian instance
		 */
		Cartesian(const Cartesian& other) noexcept = default;

		/**
		 * @brief Move Constructor
		 *
		 * @param: other: the other Cartesian instance
		 */
		Cartesian(Cartesian&& other) noexcept = default;

		/**
		 * @brief Destructor
		 */
		virtual ~Cartesian() = default;

		/**
		 * @brief Copy Assignement Operator
		 *
		 * @param: other: the other Cartesian instance
		 */
		Cartesian& operator=(const Cartesian& other) noexcept = default;

		/**
		 * @brief Move Assignement Operator
		 *
		 * @param: other: the other Cartesian instance
		 */
		Cartesian& operator=(Cartesian&& other) noexcept = default;

		/**
		 * @brief x getter
		 *
		 * @return x
		 */
		[[nodiscard]] double x() const noexcept
		{
			return x_;
		}

		/**
		 * @brief z setter
		 *
		 * @param z: z value
		 */
		void setX(double x) noexcept
		{
			x_ = x;
		}

		/**
		 * @brief y getter
		 *
		 * @return y
		 */
		[[nodiscard]] double y() const noexcept
		{
			return y_;
		}

		/**
		 * @brief y setter
		 *
		 * @param y: y value
		 */
		void setY(double y) noexcept
		{
			y_ = y;
		}

		/**
		 * @brief z getter
		 *
		 * @return z
		 */
		[[nodiscard]] double z() const noexcept
		{
			return z_;
		}

		/**
		 * @brief z setter
		 *
		 * @param z: z value
		 */
		void setZ(double z) noexcept
		{
			z_ = z;
		}

		/**
		 * @brief x Unit Vector
		 *
		 * @returns unit vector in x direction
		 */
		[[nodiscard]] static Cartesian xHat() noexcept
		{
			return Cartesian{1, 0, 0};
		}

		/**
		 * @brief y Unit Vector
		 *
		 * @returns unit vector in y direction
		 */
		[[nodiscard]] static Cartesian yHat() noexcept
		{
			return Cartesian{0, 1, 0};
		}

		/**
		 * @brief z Unit Vector
		 *
		 * @returns unit vector in z direction
		 */
		[[nodiscard]] static Cartesian zHat() noexcept
		{
			return Cartesian{0, 0, 1};
		}

		/**
		 * @brief Creates a  unit-vector in NED Cartesian coordinates from azimuth and elevation.
		 *
		 * @param azimuth: The azimuth spherical coordinate.
		 * @param elevation: The elevation spherical coordinate.
		 *
		 * @returns unit vector in pointed towards the given azimuth/elevation.
		 */
		static Cartesian nedFromAzEl(double azimuth, double elevation) noexcept
		{
			double cel = cos(elevation);
			return Cartesian(cel * cos(azimuth), cel * sin(azimuth), sin(-elevation));
		}

		/**
		 * @brief Non-Equality Operator
		 *
		 * @param other: other object
		 * @return bool true if not equal equal
		 */
		bool operator==(const Cartesian& other) const noexcept
		{
			return x_ == other.x_ && y_ == other.y_ && z_ == other.z_;
		}

		/**
		 * @brief Non-Equality Operator
		 *
		 * @param other: other object
		 * @return bool true if not equal equal
		 */
		bool operator!=(const Cartesian& other) const noexcept
		{
			return !(*this == other);
		}

	protected:
		double x_{0};
		double y_{0};
		double z_{0};
	};

	/**
	 * @brief Addition of two cartesian points
	 *
	 * @param: lhs: the left hand side object
	 * @param: rhs: the right hand side object
	 */
	[[nodiscard]] inline Cartesian operator+(const Cartesian& lhs, const Cartesian& rhs) noexcept
	{
		return Cartesian{lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z()};
	}

	/**
	 * @brief Subtraction of two cartesian points
	 *
	 * @param: lhs: the left hand side object
	 * @param: rhs: the right hand side object
	 */
	[[nodiscard]] inline Cartesian operator-(const Cartesian& lhs, const Cartesian& rhs) noexcept
	{
		return Cartesian{lhs.x() - rhs.x(), lhs.y() - rhs.y(), lhs.z() - rhs.z()};
	}

	/**
	 * @brief Dot product of two cartesian points
	 *
	 * @param: lhs: the left hand side object
	 * @param: rhs: the right hand side object
	 */
	[[nodiscard]] inline double operator*(const Cartesian& lhs, const Cartesian& rhs) noexcept
	{
		return lhs.x() * rhs.x() + lhs.y() * rhs.y() + lhs.z() * rhs.z();
	}

	/**
	 * @brief Vector scalar multiplication
	 *
	 * @param: scalar: the the scalar value
	 * @param: vec: the cartesian vector
	 */
	[[nodiscard]] inline Cartesian operator*(double scalar, const Cartesian& vec) noexcept
	{
		return Cartesian(vec.x() * scalar, vec.y() * scalar, vec.z() * scalar);
	}

	/**
	 * @brief Vector scalar multiplication
	 *
	 * @param: vec: the cartesian vector
	 * @param: scalar: the the scalar value
	 */
	[[nodiscard]] inline Cartesian operator*(const Cartesian& vec, double scalar) noexcept
	{
		return scalar * vec;
	}

	/**
	 * @brief Scalar Division a cartesian point
	 *
	 * @param: vec: the cartesian vector
	 * @param: denominator: the the scalar value
	 */
	[[nodiscard]] inline Cartesian operator/(const Cartesian& vec, double denominator) noexcept
	{
		return vec * (1.0 / denominator);
	}

	/**
	 * @brief Stream operator
	 *
	 * @param: os: the output stream
	 * @param: vec: the cartesian vector
	 */
	inline std::ostream& operator<<(std::ostream& os, const Cartesian& vec)
	{
		os << "Cartesian(x=" << vec.x() << ", y=" << vec.y() << ", z=" << vec.z() << ")\n";
		return os;
	}

	/**
	 * @brief Vector cross product
	 *
	 * @param: vec1: cartesian vector
	 * @param: vec2: cartesian vector
	 * @returns: the vector cross product
	 */
	[[nodiscard]] inline Cartesian Cross(const Cartesian& vec1, const Cartesian& vec2) noexcept
	{
		return Cartesian{vec1.y() * vec2.z() - vec1.z() * vec2.y(), -(vec1.x() * vec2.z() - vec1.z() * vec2.x()),
						 vec1.x() * vec2.y() - vec1.y() * vec2.x()};
	}

	/**
	 * @brief Vector norm
	 *
	 * @param: vec: the cartesian vector
	 * @returns: the vector norm
	 */
	[[nodiscard]] inline double Norm(const Cartesian& vec) noexcept
	{
		return std::hypot(vec.x(), vec.y(), vec.z());
	}

	/**
	 * @brief Normalize the input vector
	 *
	 * @param: vec: the cartesian vector
	 * @returns: normalized vector
	 */
	[[nodiscard]] inline Cartesian Normalize(const Cartesian& vec) noexcept
	{
		return vec / Norm(vec);
	}
} // namespace chimera::algs::common::utils

// UNCLASSIFIED
