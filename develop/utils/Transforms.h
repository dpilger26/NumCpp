// UNCLASSIFIED
#pragma once

#include <boost/math/special_functions/sign.hpp>
#include <cmath>
#include <iostream>

#include "algs/common/Constants.h"
#include "algs/common/types/Euler.h"
#include "algs/common/types/Matrix.h"
#include "algs/common/types/Orientation.h"
#include "algs/common/types/Point2dInertial.h"
#include "algs/common/types/Quaternion.h"
#include "algs/common/types/Vec3.h"
#include "algs/common/utils/Cartesian.h"
#include "algs/common/utils/Sqr.h"
#include "algs/common/utils/Wrap.h"

namespace chimera::algs::common::utils
{
	namespace reference_frames
	{
		/**
		 * @brief Geodetic coordinates
		 */
		class LLA
		{
		public:
			/**
			 * @brief Default Constructor
			 */
			LLA() = default;

			/**
			 * @brief Constructor
			 * @param latitude: latitude value
			 * @param longitude: longitude value
			 * @param altitude: altitude value
			 */
			// NOTLINTNEXTLINE(bugprone-easily-swappable-parameters)
			constexpr LLA(double latitude, double longitude, double altitude) noexcept
				: latitude_(latitude), longitude_(longitude), altitude_(altitude)
			{
			}

			/**
			 * @brief Non-Equality Operator
			 *
			 * @param other: other object
			 * @return bool true if not equal equal
			 */
			bool operator==(const LLA& other) const noexcept
			{
				return latitude_ == other.latitude_ && longitude_ == other.longitude_ && altitude_ == other.altitude_;
			}

			/**
			 * @brief Non-Equality Operator
			 *
			 * @param other: other object
			 * @return bool true if not equal equal
			 */
			bool operator!=(const LLA& other) const noexcept
			{
				return !(*this == other);
			}

			/**
			 * @brief latitude getter
			 *
			 * @return latitude
			 */
			[[nodiscard]] double latitude() const noexcept
			{
				return latitude_;
			}

			/**
			 * @brief latitude setter
			 *
			 * @param latitude: latitude value
			 */
			void setLatitude(double latitude) noexcept
			{
				latitude_ = latitude;
			}

			/**
			 * @brief longitude getter
			 *
			 * @return longitude
			 */
			[[nodiscard]] double longitude() const noexcept
			{
				return longitude_;
			}

			/**
			 * @brief longitude setter
			 *
			 * @param longitude: longitude value
			 */
			void setLongitude(double longitude) noexcept
			{
				longitude_ = longitude;
			}

			/**
			 * @brief altitude getter
			 *
			 * @return altitude
			 */
			[[nodiscard]] double altitude() const noexcept
			{
				return altitude_;
			}

			/**
			 * @brief altitude setter
			 *
			 * @param altitude: altitude value
			 */
			void setAltitude(double altitude) noexcept
			{
				altitude_ = altitude;
			}

		private:
			double latitude_{0};  // radians
			double longitude_{0}; // radians
			double altitude_{0};  // meters
		};

		/**
		 * @brief Stream operator
		 *
		 * @param: os: the output stream
		 * @param: point: the LLA point
		 */
		inline std::ostream& operator<<(std::ostream& os, const LLA& point)
		{
			os << "LLA(latitude=" << point.latitude() << ", longitude=" << point.longitude() << ", altitude=" << point.altitude() << ")\n";
			return os;
		}

		/**
		 * @brief ECEF coordinates
		 */
		class ECEF final : public Cartesian
		{
		public:
			using Cartesian::Cartesian;

			/**
			 * @brief Constructor
			 * @param cartesian: cartesian vector
			 */
			ECEF(const Cartesian& cartesian) noexcept : Cartesian(cartesian)
			{
			}
		};

		/**
		 * @brief East North Up coordinates
		 */
		class ENU final : public Cartesian
		{
		public:
			using Cartesian::Cartesian;

			/**
			 * @brief Constructor
			 * @param cartesian: cartesian vector
			 */
			ENU(const Cartesian& cartesian) noexcept : Cartesian(cartesian)
			{
			}

			/**
			 * @brief Constructor
			 * @param east: east value
			 * @param north: north value
			 * @param up: up value
			 */
			// NOTLINTNEXTLINE(bugprone-easily-swappable-parameters)
			constexpr ENU(double east, double north, double up) noexcept : Cartesian(east, north, up)
			{
			}

			/**
			 * @brief east getter
			 *
			 * @return east
			 */
			[[nodiscard]] double east() const noexcept
			{
				return x_;
			}

			/**
			 * @brief east setter
			 *
			 * @param east: east value
			 */
			void setEast(double east) noexcept
			{
				x_ = east;
			}

			/**
			 * @brief north getter
			 *
			 * @return double
			 */
			[[nodiscard]] double north() const noexcept
			{
				return y_;
			}

			/**
			 * @brief north setter
			 *
			 * @param north: north value
			 */
			void setNorth(double north) noexcept
			{
				y_ = north;
			}

			/**
			 * @brief up getter
			 *
			 * @return up
			 */
			[[nodiscard]] double up() const noexcept
			{
				return z_;
			}

			/**
			 * @brief up setter
			 *
			 * @param up: up value
			 */
			void setUp(double up) noexcept
			{
				z_ = up;
			}
		};

		/**
		 * @brief Stream operator
		 *
		 * @param: os: the output stream
		 * @param: point: the ENU point
		 */
		inline std::ostream& operator<<(std::ostream& os, const ENU& point)
		{
			os << "ENU(east=" << point.east() << ", north=" << point.north() << ", up=" << point.up() << ")\n";
			return os;
		}

		/**
		 * @brief North east down coordinates
		 */
		class NED final : public Cartesian
		{
		public:
			using Cartesian::Cartesian;

			/**
			 * @brief Constructor
			 * @param cartesian: cartesian vector
			 */
			NED(const Cartesian& cartesian) noexcept : Cartesian(cartesian)
			{
			}

			/**
			 * @brief Constructor
			 * @param north: north value
			 * @param east: east value
			 * @param down: down value
			 */
			// NOTLINTNEXTLINE(bugprone-easily-swappable-parameters)
			constexpr NED(double north, double east, double down) noexcept : Cartesian(north, east, down)
			{
			}

			/**
			 * @brief north getter
			 *
			 * @return north
			 */
			[[nodiscard]] double north() const noexcept
			{
				return x_;
			}

			/**
			 * @brief north setter
			 *
			 * @param north: north value
			 */
			void setNorth(double north) noexcept
			{
				x_ = north;
			}

			/**
			 * @brief east getter
			 *
			 * @return double
			 */
			[[nodiscard]] double east() const noexcept
			{
				return y_;
			}

			/**
			 * @brief east setter
			 *
			 * @param east: east value
			 */
			void setEast(double east) noexcept
			{
				y_ = east;
			}

			/**
			 * @brief down getter
			 *
			 * @return down
			 */
			[[nodiscard]] double down() const noexcept
			{
				return z_;
			}

			/**
			 * @brief down setter
			 *
			 * @param down: down value
			 */
			void setDown(double down) noexcept
			{
				z_ = down;
			}
		};

		/**
		 * @brief Az El coordinates
		 */

		class AzEl
		{
		public:
			/**
			 * @brief Default Constructor
			 */
			AzEl() = default;

			/**
			 * @brief Constructor
			 * @param az: az value
			 * @param el: el value
			 */
			// NOTLINTNEXTLINE(bugprone-easily-swappable-parameters)
			constexpr AzEl(double az, double el) noexcept : az_(az), el_(el)
			{
			}

			/**
			 * @brief az getter
			 *
			 * @return az
			 */
			[[nodiscard]] double az() const noexcept
			{
				return az_;
			}

			/**
			 * @brief az setter
			 *
			 * @param az: az value
			 */
			void setAz(double az) noexcept
			{
				az_ = az;
			}

			/**
			 * @brief el getter
			 *
			 * @return el
			 */
			[[nodiscard]] double el() const noexcept
			{
				return el_;
			}

			/**
			 * @brief el setter
			 *
			 * @param el: el value
			 */
			void setEl(double el) noexcept
			{
				el_ = el;
			}

			/**
			 * @brief Non-Equality Operator
			 *
			 * @param other: other object
			 * @return bool true if not equal equal
			 */
			bool operator==(const AzEl& other) const noexcept
			{
				return az_ == other.az_ && el_ == other.el_;
			}

			/**
			 * @brief Non-Equality Operator
			 *
			 * @param other: other object
			 * @return bool true if not equal equal
			 */
			bool operator!=(const AzEl& other) const noexcept
			{
				return !(*this == other);
			}

		private:
			double az_{0}; // radians
			double el_{0}; // radians
		};

		/**
		 * @brief Stream operator
		 *
		 * @param: os: the output stream
		 * @param: point: the AzEl point
		 */
		inline std::ostream& operator<<(std::ostream& os, const AzEl& point)
		{
			os << "AzEl(az=" << point.az() << ", el=" << point.el() << ")\n";
			return os;
		}
	} // namespace reference_frames

	namespace transforms
	{
		/**
		 * @brief Converts the LLA coordinates to ECEF
		 *        https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates
		 *
		 * @param point: the point of interest
		 * @returns Cartesian
		 */
		[[nodiscard]] inline reference_frames::ECEF LLAtoECEF(const reference_frames::LLA& point) noexcept
		{
			constexpr auto B2_DIV_A2 = Sqr(constants::EARTH_POLAR_RADIUS / constants::EARTH_EQUATORIAL_RADIUS);
			constexpr auto E_SQR = 1. - B2_DIV_A2;

			const auto sinLat = std::sin(point.latitude());
			const auto cosLat = std::cos(point.latitude());
			const auto sinLon = std::sin(point.longitude());
			const auto cosLon = std::cos(point.longitude());

			// prime vertical meridian
			const auto pvm = constants::EARTH_EQUATORIAL_RADIUS / std::sqrt(1. - E_SQR * Sqr(sinLat));

			return reference_frames::ECEF{(pvm + point.altitude()) * cosLat * cosLon, (pvm + point.altitude()) * cosLat * sinLon,
										  (B2_DIV_A2 * pvm + point.altitude()) * sinLat};
		}

		/**
		 * @brief Converts ECEF coordinates to LLA
		 * 		  https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_geodetic_coordinates
		 *
		 * @param ecef the point of interest
		 * @param tol Tolerance for the convergence of altitude (overriden if 10 iterations are processed)
		 * @return reference_frames::LLA
		 */
		[[nodiscard]] inline reference_frames::LLA ECEFtoLLA(const reference_frames::ECEF& ecef, double tol = 1e-8) noexcept
		{
			constexpr int MAX_ITER = 10;
			constexpr auto E_SQR = 1 - Sqr(constants::EARTH_POLAR_RADIUS / constants::EARTH_EQUATORIAL_RADIUS);

			const auto p = std::hypot(ecef.x(), ecef.y());
			const auto lon = std::atan2(ecef.y(), ecef.x());

			double alt = 0.0;
			double lat = 0.0;

			if(p < tol)
			{
				lat = boost::math::sign(ecef.z()) * boost::math::constants::half_pi<double>();
				alt = std::abs(ecef.z()) - constants::EARTH_POLAR_RADIUS;
			}
			else
			{
				// Iteratively update latitude and altitude.
				// This is expected to converge in ~4 iterations, but apply a maximum number of iterations incase tol is too small
				double err = 1.0;
				int iter = 0;
				while(err > tol && iter < MAX_ITER)
				{
					double N = constants::EARTH_EQUATORIAL_RADIUS / std::sqrt(1 - E_SQR * Sqr(std::sin(lat)));
					lat = std::atan((ecef.z() / p) / (1 - (N * E_SQR / (N + alt))));
					double newAlt = (p / std::cos(lat)) - N;
					err = std::abs(alt - newAlt);
					alt = newAlt;
					iter++;
				}
			}
			return reference_frames::LLA{lat, lon, alt};
		}

		/**
		 * @brief Converts NED to ENU
		 *
		 * @param ned: the NED coordinates
		 * @returns ENU
		 */
		[[nodiscard]] inline reference_frames::ENU NEDtoENU(const reference_frames::NED& point) noexcept
		{
			return {point.east(), point.north(), -point.down()};
		}

		/**
		 * @brief Converts ENU to NED
		 *
		 * @param enu: the ENU coordinates
		 * @returns NED
		 */
		[[nodiscard]] inline reference_frames::NED ENUtoNED(const reference_frames::ENU& point) noexcept
		{
			return {point.north(), point.east(), -point.up()};
		}

		/**
		 * @brief Converts the ECEF coordinates to ENU
		 *        https://apps.dtic.mil/sti/pdfs/AD1170763.pdf
		 *        Figure 11 https://apps.dtic.mil/sti/pdfs/AD1170763.pdf for a helpful diagram
		 *
		 * @param target: the target of interest
		 * @returns ENU
		 */
		[[nodiscard]] inline reference_frames::ENU ECEFtoENU(const reference_frames::ECEF& target, const reference_frames::LLA& platform) noexcept
		{
			const auto sinLat = std::sin(platform.latitude());
			const auto cosLat = std::cos(platform.latitude());
			const auto sinLon = std::sin(platform.longitude());
			const auto cosLon = std::cos(platform.longitude());

			const auto platformECEF = LLAtoECEF(platform);

			const auto x = target.x() - platformECEF.x();
			const auto y = target.y() - platformECEF.y();
			const auto z = target.z() - platformECEF.z();

			return reference_frames::ENU{-sinLon * x + cosLon * y, -sinLat * cosLon * x - sinLat * sinLon * y + cosLat * z,
										 cosLat * cosLon * x + cosLat * sinLon * y + sinLat * z};
		}

		/**
		 * @brief Converts the ECEF coordinates to ENU
		 *        https://apps.dtic.mil/sti/pdfs/AD1170763.pdf
		 *        Figure 11 https://apps.dtic.mil/sti/pdfs/AD1170763.pdf for a helpful diagram
		 *
		 * @param target: the target of interest
		 * @returns ENU
		 */
		[[nodiscard]] inline reference_frames::ENU ECEFtoENU(const reference_frames::ECEF& target, const reference_frames::ECEF& platform) noexcept
		{
			return ECEFtoENU(target, ECEFtoLLA(platform));
		}

		/**
		 * @brief Converts the ECEF coordinates to NED
		 *        https://apps.dtic.mil/sti/pdfs/AD1170763.pdf
		 *        Figure 11 https://apps.dtic.mil/sti/pdfs/AD1170763.pdf for a helpful diagram
		 *
		 * @param target: the target of interest
		 * @param platform: the platform location
		 * @returns NED
		 */
		[[nodiscard]] inline reference_frames::NED ECEFtoNED(const reference_frames::ECEF& target, const reference_frames::LLA& platform) noexcept
		{
			return ENUtoNED(ECEFtoENU(target, platform));
		}

		/**
		 * @brief Converts the ECEF coordinates to NED
		 *        https://apps.dtic.mil/sti/pdfs/AD1170763.pdf
		 *        Figure 11 https://apps.dtic.mil/sti/pdfs/AD1170763.pdf for a helpful diagram
		 *
		 * @param target: the target of interest
		 * @param platform: the platform location
		 * @returns NED
		 */
		[[nodiscard]] inline reference_frames::NED ECEFtoNED(const reference_frames::ECEF& target, const reference_frames::ECEF& platform) noexcept
		{
			return ENUtoNED(ECEFtoENU(target, platform));
		}

		/**
		 * @brief Converts the NED coordinates to ECEF
		 *        https://apps.dtic.mil/sti/pdfs/AD1170763.pdf
		 *        Figure 11 https://apps.dtic.mil/sti/pdfs/AD1170763.pdf for a helpful diagram
		 *        https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates
		 *
		 * @param target: the target of interest
		 * @param platform: the platform location
		 * @returns NED
		 */
		[[nodiscard]] inline reference_frames::ECEF NEDtoECEF(const reference_frames::NED& target, const reference_frames::ECEF& platform) noexcept
		{
			const auto platformLLA = ECEFtoLLA(platform);

			const auto sinLat = std::sin(platformLLA.latitude());
			const auto cosLat = std::cos(platformLLA.latitude());
			const auto sinLon = std::sin(platformLLA.longitude());
			const auto cosLon = std::cos(platformLLA.longitude());

			auto rotationMatrix = types::Matrix<double, 3, 3>{};
			rotationMatrix(0, 0) = -sinLat * cosLon;
			rotationMatrix(1, 0) = -sinLat * sinLon;
			rotationMatrix(2, 0) = cosLat;
			rotationMatrix(0, 1) = -sinLon;
			rotationMatrix(1, 1) = cosLon;
			rotationMatrix(2, 1) = 0.;
			rotationMatrix(0, 2) = -cosLat * cosLon;
			rotationMatrix(1, 2) = -cosLat * sinLon;
			rotationMatrix(2, 2) = -sinLat;

			auto targetVec = types::Vector<double, 3>{};
			targetVec[0] = target.north();
			targetVec[1] = target.east();
			targetVec[2] = target.down();

			auto platformVec = types::Vector<double, 3>{};
			platformVec[0] = platform.x();
			platformVec[1] = platform.y();
			platformVec[2] = platform.z();

			const auto targetECEFVec = (rotationMatrix * targetVec) + platformVec;
			return {targetECEFVec[0], targetECEFVec[1], targetECEFVec[2]};
		}

		/**
		 * @brief Converts the NED coordinates to ECEF
		 *        https://apps.dtic.mil/sti/pdfs/AD1170763.pdf
		 *        Figure 11 https://apps.dtic.mil/sti/pdfs/AD1170763.pdf for a helpful diagram
		 *        https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates
		 *
		 * @param target: the target of interest
		 * @param platform: the platform location
		 * @returns NED
		 */
		[[nodiscard]] inline reference_frames::ECEF NEDtoECEF(const reference_frames::NED& target, const reference_frames::LLA& platform) noexcept
		{
			return NEDtoECEF(target, LLAtoECEF(platform));
		}

		/**
		 * @brief Converts the NED coordinates to LLA
		 *        https://apps.dtic.mil/sti/pdfs/AD1170763.pdf
		 *        Figure 11 https://apps.dtic.mil/sti/pdfs/AD1170763.pdf for a helpful diagram
		 *        https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates
		 *
		 * @param target: the target of interest
		 * @param platform: the platform location
		 * @returns NED
		 */
		[[nodiscard]] inline reference_frames::LLA NEDtoLLA(const reference_frames::NED& target, const reference_frames::ECEF& platform) noexcept
		{
			return ECEFtoLLA(NEDtoECEF(target, platform));
		}

		/**
		 * @brief Converts the NED coordinates to LLA
		 *        https://apps.dtic.mil/sti/pdfs/AD1170763.pdf
		 *        Figure 11 https://apps.dtic.mil/sti/pdfs/AD1170763.pdf for a helpful diagram
		 *        https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates
		 *
		 * @param target: the target of interest
		 * @param platform: the platform location
		 * @returns NED
		 */
		[[nodiscard]] inline reference_frames::LLA NEDtoLLA(const reference_frames::NED& target, const reference_frames::LLA& platform) noexcept
		{
			return ECEFtoLLA(NEDtoECEF(target, platform));
		}

		/**
		 * @brief Converts the LLA coordinates to Az El with geocentric up
		 *        https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
		 *        Figure 11 https://apps.dtic.mil/sti/pdfs/AD1170763.pdf for a helpful diagram
		 *
		 * @param target: the target of interest
		 * @param platform: the platform
		 * @returns AzEl
		 */
		[[nodiscard]] inline reference_frames::AzEl ECEFtoAzElGeocentric(const reference_frames::ECEF& target,
																		 const reference_frames::ECEF& platform) noexcept
		{
			const auto rhoHat = Normalize(target - platform);
			const auto uHat = Normalize(platform);
			const auto eHat = Normalize(Cross(Cartesian::zHat(), uHat));
			const auto nHat = Normalize(Cross(uHat, eHat));

			return reference_frames::AzEl{Wrap2Pi(std::atan2(rhoHat * eHat, rhoHat * nHat)), std::asin(rhoHat * uHat)};
		}

		/**
		 * @brief Converts the LLA coordinates to Az El with geocentric up
		 *        https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
		 *        Figure 11 https://apps.dtic.mil/sti/pdfs/AD1170763.pdf for a helpful diagram
		 *
		 * @param target: the target of interest
		 * @param platform: the platform
		 * @returns AzEl
		 */
		[[nodiscard]] inline reference_frames::AzEl LLAtoAzElGeocentric(const reference_frames::LLA& target,
																		const reference_frames::LLA& platform) noexcept
		{
			return ECEFtoAzElGeocentric(LLAtoECEF(target), LLAtoECEF(platform));
		}

		/**
		 * @brief Converts the LLA coordinates to Az El with geodedic up
		 *        https://geospace-code.github.io/matmap3d/enu2aer.html
		 *
		 * @param target: the target of interest
		 * @param platform: the platform
		 * @returns AzEl
		 */
		[[nodiscard]] inline reference_frames::AzEl ECEFtoAzElGeodetic(const reference_frames::ECEF& target,
																	   const reference_frames::LLA& platform) noexcept
		{
			const auto targetENU = ECEFtoENU(target, platform);
			const auto targetENUNormalizedCart = Normalize(Cartesian{targetENU.east(), targetENU.north(), targetENU.up()});
			const auto& east = targetENUNormalizedCart.x();
			const auto& north = targetENUNormalizedCart.y();
			const auto& up = targetENUNormalizedCart.z();

			return reference_frames::AzEl{Wrap2Pi(std::atan2(east, north)), std::asin(up)};
		}

		/**
		 * @brief Converts the LLA coordinates to Az El with geodedic up
		 *        https://geospace-code.github.io/matmap3d/enu2aer.html
		 *
		 * @param target: the target of interest
		 * @param platform: the platform
		 * @returns AzEl
		 */
		[[nodiscard]] inline reference_frames::AzEl ECEFtoAzElGeodetic(const reference_frames::ECEF& target,
																	   const reference_frames::ECEF& platform) noexcept
		{
			return ECEFtoAzElGeodetic(target, ECEFtoLLA(platform));
		}

		/**
		 * @brief Converts the LLA coordinates to Az El with geodedic up
		 *        https://geospace-code.github.io/matmap3d/enu2aer.html
		 *
		 * @param target: the target of interest
		 * @param platform: the platform
		 * @returns AzEl
		 */
		[[nodiscard]] inline reference_frames::AzEl LLAtoAzElGeodetic(const reference_frames::LLA& target,
																	  const reference_frames::LLA& platform) noexcept
		{
			return ECEFtoAzElGeodetic(LLAtoECEF(target), platform);
		}

		/**
		 * @brief Converts the Cartesian XYZ (NED) coordinates to 2d speherical inertial coordinates.
		 *        Range is not used.
		 *        NOTE: positive elevation is defined as the negative z (up) direction
		 *
		 * @param cartesian: coordinates to convert
		 * @returns Point2dInertial
		 */
		[[nodiscard]] inline types::Point2dInertial CartToInertial(const Cartesian& cartesian) noexcept
		{
			const auto hypotXy = std::hypot(cartesian.x(), cartesian.y());
			const auto elAxis = types::InertialAxis(-std::atan2(cartesian.z(), hypotXy));
			const auto azAxis = types::InertialAxis(Wrap2Pi(std::atan2(cartesian.y(), cartesian.x())));
			auto point = types::Point2dInertial();
			point.setAz(azAxis);
			point.setEl(elAxis);
			return point;
		}

		/**
		 * @brief Converts the spherical inertial coordinates (NED) to Cartesian XYZ (NED).
		 *        NOTE: positive elevation is defined as the negative z (up) direction
		 *
		 * @param inertial 2D Inertial point with azimuth and elevation
		 * @param range Optional range
		 * @return NED
		 */
		[[nodiscard]] inline reference_frames::NED InertialToCart(const types::Point2dInertial inertial, double range = 1.0) noexcept
		{
			const auto north = range * std::cos(-inertial.el().position()) * std::cos(inertial.az().position());
			const auto east = range * std::cos(-inertial.el().position()) * std::sin(inertial.az().position());
			const auto down = range * std::sin(-inertial.el().position());
			return reference_frames::NED{north, east, down};
		}

		namespace detail
		{
			/**
			 * @brief get the local NED unit vectors wrt the ECEF coordinate system
			 *        // https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
			 *
			 * @param latitude: the latitude in radians
			 * @param longitude: the longitude in radians
			 * @return std::array<types::Vec3, 3>
			 */
			[[nodiscard]] inline std::array<types::Vec3, 3> NEDUnitVecsInECEF(const reference_frames::ECEF& location) noexcept
			{
				const auto lla = ECEFtoLLA(location);

				const auto sinLat = std::sin(lla.latitude());
				const auto cosLat = std::cos(lla.latitude());
				const auto sinLon = std::sin(lla.longitude());
				const auto cosLon = std::cos(lla.longitude());

				const auto xHat = types::Vec3{-cosLon * sinLat, -sinLon * sinLat, cosLat};
				const auto yHat = types::Vec3{-sinLon, cosLon, 0.};
				const auto zHat = types::Vec3{-cosLon * cosLat, -sinLon * cosLat, -sinLat};

				return {xHat, yHat, zHat};
			}
		}

		/**
		 * @brief Converts ECEF euler angles to body roll/pitch/yaw
		 *
		 * @param location: the ecef location
		 * @param orientation: ecef euler angles
		 * @return Orientation
		 */
		[[nodiscard]] inline types::Orientation ECEFEulerToNEDRollPitchYaw(const reference_frames::ECEF& location,
																		   const types::Euler& orientation) noexcept
		{
			const auto x0 = types::Vec3::right();
			const auto y0 = types::Vec3::up();
			const auto z0 = types::Vec3::forward();

			// first rotation array, z0 by psi
			const auto quatPsi = types::Quaternion{z0, orientation.psi()};

			// rotate
			const auto x1 = quatPsi * x0;
			const auto y1 = quatPsi * y0;

			// second rotation array, y1 by theta
			const auto quatTheta = types::Quaternion{y1, orientation.theta()};

			// rotate
			const auto x2 = quatTheta * x1;
			const auto y2 = quatTheta * y1;

			// third rotation array, x2 by phi
			const auto quatPhi = types::Quaternion{x2, orientation.phi()};

			// rotate
			const auto x3 = quatPhi * x2;
			const auto y3 = quatPhi * y2;

			// get the local NED unit vectors wrt the ECEF coordinate system
			const auto& [xHat0, yHat0, zHat0] = detail::NEDUnitVecsInECEF(location);

			// calculate yaw and pitch
			const auto yaw = std::atan2(x3.dot(yHat0), x3.dot(xHat0));
			const auto pitch = std::atan(-x3.dot(zHat0) / std::hypot(x3.dot(xHat0), x3.dot(yHat0)));

			// calculate roll
			const auto yHat2 = (types::Quaternion{zHat0, yaw} * yHat0);
			const auto zHat2 = (types::Quaternion{yHat2, pitch} * zHat0);
			const auto roll = std::atan2(y3.dot(zHat2), y3.dot(yHat2));

			return {utils::Wrap(yaw), pitch, utils::Wrap(roll)};
		}

		/**
		 * @brief Converts ECEF euler angles to body roll/pitch/yaw
		 *
		 * @param location: the ecef location
		 * @param orientation: ned euler angles
		 * @return Orientation
		 */
		[[nodiscard]] inline types::Euler NEDRollPitchYawToECEFEuler(const reference_frames::ECEF& location,
																	 const types::Orientation& orientation) noexcept
		{
			// get the local NED unit vectors wrt the ECEF coordinate system
			const auto& [x0, y0, z0] = detail::NEDUnitVecsInECEF(location);

			// first rotation array, z0 by yaw
			const auto quatYaw = types::Quaternion{z0, orientation.yaw()};

			// rotate
			const auto x1 = quatYaw * x0;
			const auto y1 = quatYaw * y0;

			// second rotation array, y1 by pitch
			const auto quatPitch = types::Quaternion{y1, orientation.pitch()};

			// rotate
			const auto x2 = quatPitch * x1;
			const auto y2 = quatPitch * y1;

			// third rotation array, x2 by roll
			const auto quatRoll = types::Quaternion{x2, orientation.roll()};

			// rotate
			const auto x3 = quatRoll * x2;
			const auto y3 = quatRoll * y2;

			// calculate phi and theta
			const auto xHat0 = types::Vec3::right();
			const auto yHat0 = types::Vec3::up();
			const auto zHat0 = types::Vec3::forward();

			const auto psi = std::atan2(x3.dot(yHat0), x3.dot(xHat0));
			const auto theta = std::atan(-x3.dot(zHat0) / std::hypot(x3.dot(xHat0), x3.dot(yHat0)));

			// calculate phi
			const auto yHat2 = (types::Quaternion{zHat0, psi} * yHat0);
			const auto zHat2 = (types::Quaternion{yHat2, theta} * zHat0);
			const auto phi = std::atan2(y3.dot(zHat2), y3.dot(yHat2));

			return {utils::Wrap(psi), theta, utils::Wrap(phi)};
		}
	} // namespace transforms
} // namespace chimera::algs::common::utils

// UNCLASSIFIED
