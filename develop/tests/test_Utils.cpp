// UNCLASSIFIED

#include "gtest/gtest.h"
#include "test.h"

#include <cmath>
#include <limits>
#include <random>

#include "boost/math/constants/constants.hpp"

#include "ball/common/utils/Math.h"

using namespace ball::common::utils::math::types;
using namespace ball::common::utils::math::utils;
using namespace std::chrono_literals;

namespace ball::common::utils::math
{
	constexpr auto eps = std::numeric_limits<double>::epsilon();

	class UtilsTestSuite : public BaseTest
	{
	};

	/**
	 * @brief test Cartesian
	 */
	TEST_F(UtilsTestSuite, TestCartesian)
	{
		auto vec = Cartesian{};
		ASSERT_NEAR(vec.x(), 0.0, eps);
		ASSERT_NEAR(vec.y(), 0.0, eps);
		ASSERT_NEAR(vec.z(), 0.0, eps);

		vec = Cartesian{1, 2, 3};
		ASSERT_NEAR(vec.x(), 1.0, eps);
		ASSERT_NEAR(vec.y(), 2.0, eps);
		ASSERT_NEAR(vec.z(), 3.0, eps);

		const auto vecX = Cartesian::xHat();
		ASSERT_NEAR(vecX.x(), 1.0, eps);
		ASSERT_NEAR(vecX.y(), 0.0, eps);
		ASSERT_NEAR(vecX.z(), 0.0, eps);

		const auto vecY = Cartesian::yHat();
		ASSERT_NEAR(vecY.x(), 0.0, eps);
		ASSERT_NEAR(vecY.y(), 1.0, eps);
		ASSERT_NEAR(vecY.z(), 0.0, eps);

		const auto vecZ = Cartesian::zHat();
		ASSERT_NEAR(vecZ.x(), 0.0, eps);
		ASSERT_NEAR(vecZ.y(), 0.0, eps);
		ASSERT_NEAR(vecZ.z(), 1.0, eps);

		const auto vecAdd = vecX + vecY + vecZ;
		ASSERT_NEAR(vecAdd.x(), 1.0, eps);
		ASSERT_NEAR(vecAdd.y(), 1.0, eps);
		ASSERT_NEAR(vecAdd.z(), 1.0, eps);

		const auto vecSub = vecX - vecY - vecZ;
		ASSERT_NEAR(vecSub.x(), 1.0, eps);
		ASSERT_NEAR(vecSub.y(), -1.0, eps);
		ASSERT_NEAR(vecSub.z(), -1.0, eps);

		auto vecMult = 2 * vecAdd;
		ASSERT_NEAR(vecMult.x(), 2.0, eps);
		ASSERT_NEAR(vecMult.y(), 2.0, eps);
		ASSERT_NEAR(vecMult.z(), 2.0, eps);

		vecMult = vecAdd * 2;
		ASSERT_NEAR(vecMult.x(), 2.0, eps);
		ASSERT_NEAR(vecMult.y(), 2.0, eps);
		ASSERT_NEAR(vecMult.z(), 2.0, eps);

		const auto vecDiv = vecAdd / 2;
		ASSERT_NEAR(vecDiv.x(), 0.5, eps);
		ASSERT_NEAR(vecDiv.y(), 0.5, eps);
		ASSERT_NEAR(vecDiv.z(), 0.5, eps);

		auto vecCross = Cross(vecX, vecY);
		ASSERT_NEAR(vecCross.x(), vecZ.x(), eps);
		ASSERT_NEAR(vecCross.y(), vecZ.y(), eps);
		ASSERT_NEAR(vecCross.z(), vecZ.z(), eps);

		vecCross = Cross(vecY, vecZ);
		ASSERT_NEAR(vecCross.x(), vecX.x(), eps);
		ASSERT_NEAR(vecCross.y(), vecX.y(), eps);
		ASSERT_NEAR(vecCross.z(), vecX.z(), eps);

		vecCross = Cross(vecZ, vecX);
		ASSERT_NEAR(vecCross.x(), vecY.x(), eps);
		ASSERT_NEAR(vecCross.y(), vecY.y(), eps);
		ASSERT_NEAR(vecCross.z(), vecY.z(), eps);

		const auto vecNorm = Norm(Cartesian{1, 2, 3});
		ASSERT_NEAR(vecNorm, std::hypot(1.0, 2.0, 3.0), eps);

		const auto vecNormalized = Normalize(Cartesian{1, 1, 1});
		const auto expectedValue = 1.0 / sqrt(3.0);
		ASSERT_NEAR(vecNormalized.x(), expectedValue, eps);
		ASSERT_NEAR(vecNormalized.y(), expectedValue, eps);
		ASSERT_NEAR(vecNormalized.z(), expectedValue, eps);
	}

	/**
	 * @brief test EssentiallyEqual
	 */
	TEST_F(UtilsTestSuite, TestEssentiallyEqual)
	{
		ASSERT_TRUE(EssentiallyEqual(3.1415926, 3.1415926));
		ASSERT_FALSE(EssentiallyEqual(3.1415926, 3.1415927));
	}

	/**
	 * @brief test GenerateUUID
	 */
	TEST_F(UtilsTestSuite, TestGenerateUUID)
	{
		const auto id1 = GenerateUUID();
		const auto id2 = GenerateUUID();
		ASSERT_NE(id1, id2);
	}

	/**
	 * @brief test GenerateUUIDStr
	 */
	TEST_F(UtilsTestSuite, TestGenerateUUIDStr)
	{
		const auto id1 = GenerateUUIDStr();
		const auto id2 = GenerateUUIDStr();
		ASSERT_NE(id1, id2);
	}

	/**
	 * @brief test IsTargetVisible
	 */
	TEST_F(UtilsTestSuite, TestIsTargetVisible)
	{
		const auto platform = PlatformKinematics(Point4d(0, 0, 10'000, Clock::now()));

		auto target = Point2d(0.017, 0.017);
		ASSERT_TRUE(IsTargetVisible(platform, target));

		target.setAltitude(20'000);
		ASSERT_TRUE(IsTargetVisible(platform, target));

		target.setLongitude(3.124);
		ASSERT_FALSE(IsTargetVisible(platform, target));

		auto target2 = Point4d(0.017, 0.017, 20'000, Clock::now());
		ASSERT_TRUE(IsTargetVisible(platform, target2));

		target2.setLongitude(3.124);
		ASSERT_FALSE(IsTargetVisible(platform, target2));
	}

	/**
	 * @brief test IsValid
	 *        NOTE: this is definitely not exhaustive, that would take a much larger effort.
	 */
	TEST_F(UtilsTestSuite, TestIsValid)
	{
		auto airVolumeSensorReferenced = AirVolumeSensorReferenced{};
		ASSERT_TRUE(IsValid(airVolumeSensorReferenced));
		airVolumeSensorReferenced.setMinRangeOfInterest(-20.0);
		ASSERT_FALSE(IsValid(airVolumeSensorReferenced));

		auto altitudeRange = AltitudeRange{};
		ASSERT_TRUE(IsValid(altitudeRange));
		altitudeRange.setMin(-10005);
		ASSERT_FALSE(IsValid(altitudeRange));

		auto lineOfBearingAndUncertainty = LineOfBearingAndUncertainty{};
		ASSERT_TRUE(IsValid(lineOfBearingAndUncertainty));
		lineOfBearingAndUncertainty.setLos(LOS(1000, 1000));
		ASSERT_FALSE(IsValid(lineOfBearingAndUncertainty));

		auto losOption = LOS_Option{};
		ASSERT_TRUE(IsValid(losOption));
		losOption.setAz(1000);
		ASSERT_FALSE(IsValid(losOption));

		auto los = LOS{};
		ASSERT_TRUE(IsValid(los));
		los.setBearing(1000);
		ASSERT_FALSE(IsValid(los));

		auto losUncertainty = LOSUncertainty{};
		ASSERT_TRUE(IsValid(losUncertainty));
		losUncertainty.setBearing(1000);
		ASSERT_FALSE(IsValid(losUncertainty));

		auto orientation = Orientation{};
		ASSERT_TRUE(IsValid(orientation));
		orientation.setPitch(1000);
		ASSERT_FALSE(IsValid(orientation));

		auto orientationRate = OrientationRate{};
		ASSERT_TRUE(IsValid(orientationRate));
		orientationRate.setPitch(10000);
		ASSERT_FALSE(IsValid(orientationRate));

		auto orientationAcceleration = OrientationAcceleration{};
		ASSERT_TRUE(IsValid(orientationAcceleration));
		orientationAcceleration.setPitch(10000);
		ASSERT_FALSE(IsValid(orientationAcceleration));

		auto platformKinematics = PlatformKinematics{};
		ASSERT_TRUE(IsValid(platformKinematics));
		platformKinematics.setPosition(Point4d(1000, 0, 0, Clock::now()));
		ASSERT_FALSE(IsValid(platformKinematics));

		auto point2d = Point2d{};
		ASSERT_TRUE(IsValid(point2d));
		point2d.setLatitude(1000);
		ASSERT_FALSE(IsValid(point2d));

		auto point4d = Point4d{};
		ASSERT_TRUE(IsValid(point4d));
		point4d.setLatitude(1000);
		ASSERT_FALSE(IsValid(point4d));

		auto point2dInertial = Point2dInertial{};
		ASSERT_TRUE(IsValid(point2dInertial));
		point2dInertial.setAz(InertialAxis(1000));
		ASSERT_FALSE(IsValid(point2dInertial));

		auto pointTarget = PointTarget{};
		ASSERT_TRUE(IsValid(pointTarget));
		pointTarget.setPoint(Point2d(1000, 1000));
		ASSERT_FALSE(IsValid(pointTarget));

		auto relativeSlantRange = RelativeSlantRangeLOS3d{};
		ASSERT_TRUE(IsValid(relativeSlantRange));
		relativeSlantRange.setRange(-1.);
		ASSERT_FALSE(IsValid(relativeSlantRange));
	}

	/**
	 * @brief test LOS_OptionToPoint2dInertial
	 */
	TEST_F(UtilsTestSuite, TestLOS_OptionToPoint2dInertial)
	{
		const double az = 1.0;
		const double azRate = 1.1;
		const double el = 2.0;
		const double elRate = 2.1;
		const double roll = 3.0;
		const double rollRate = 3.1;
		const auto losOption = LOS_Option(az, azRate, el, elRate, roll, rollRate);
		const auto point2dInertial = LOS_OptionToPoint2dInertial(losOption);

		ASSERT_EQ(point2dInertial.az().position(), az);
		ASSERT_EQ(point2dInertial.az().rate(), azRate);
		ASSERT_EQ(point2dInertial.el().position(), el);
		ASSERT_EQ(point2dInertial.el().rate(), elRate);
	}

	/**
	 * @brief test PropagatePlatformKinematics
	 */
	TEST_F(UtilsTestSuite, TestPropagatePlatformKinematics)
	{
		const auto now = Clock::now();
		const auto propagateTime = now + 10s;
		auto platformKinematics = PlatformKinematics{Point4d{0., 0., 0., now}};
		auto newPlatformKinematics = PropagatePlatformKinematics(platformKinematics, propagateTime);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.position().latitude(), 0.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.position().longitude(), 0.);
		ASSERT_NEAR(newPlatformKinematics.position().altitude(), 0., 1e-9);
		ASSERT_TRUE(newPlatformKinematics.position().timestamp() == propagateTime);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().north(), 0.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().east(), 0.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().down(), 0.);
		ASSERT_FALSE(newPlatformKinematics.acceleration());
		ASSERT_FALSE(newPlatformKinematics.orientation());
		ASSERT_FALSE(newPlatformKinematics.orientationRate());
		ASSERT_FALSE(newPlatformKinematics.orientationAcceleration());

		platformKinematics = PlatformKinematics{Point4d{0.1, 0.2, 10'000., now}};
		newPlatformKinematics = PropagatePlatformKinematics(platformKinematics, propagateTime);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.position().latitude(), 0.1);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.position().longitude(), 0.2);
		ASSERT_NEAR(newPlatformKinematics.position().altitude(), 10'000., 1e-9);
		ASSERT_TRUE(newPlatformKinematics.position().timestamp() == propagateTime);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().north(), 0.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().east(), 0.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().down(), 0.);
		ASSERT_FALSE(newPlatformKinematics.acceleration());
		ASSERT_FALSE(newPlatformKinematics.orientation());
		ASSERT_FALSE(newPlatformKinematics.orientationRate());
		ASSERT_FALSE(newPlatformKinematics.orientationAcceleration());

		platformKinematics = PlatformKinematics{Point4d{0.1, 0.2, 10'000., now}, Velocity3d{100., 101., 102.}};
		newPlatformKinematics = PropagatePlatformKinematics(platformKinematics, propagateTime);
		ASSERT_GT(newPlatformKinematics.position().latitude(), platformKinematics.position().latitude());
		ASSERT_GT(newPlatformKinematics.position().longitude(), platformKinematics.position().longitude());
		ASSERT_LT(newPlatformKinematics.position().altitude(), platformKinematics.position().altitude());
		ASSERT_TRUE(newPlatformKinematics.position().timestamp() == propagateTime);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().north(), 100.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().east(), 101.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().down(), 102.);
		ASSERT_FALSE(newPlatformKinematics.acceleration());
		ASSERT_FALSE(newPlatformKinematics.orientation());
		ASSERT_FALSE(newPlatformKinematics.orientationRate());
		ASSERT_FALSE(newPlatformKinematics.orientationAcceleration());

		platformKinematics = PlatformKinematics{Point4d{0.1, 0.2, 10'000., now}, Velocity3d{100., 101., 102.}, Acceleration3d{50., 51., 52.}};
		newPlatformKinematics = PropagatePlatformKinematics(platformKinematics, propagateTime);
		ASSERT_GT(newPlatformKinematics.position().latitude(), platformKinematics.position().latitude());
		ASSERT_GT(newPlatformKinematics.position().longitude(), platformKinematics.position().longitude());
		ASSERT_LT(newPlatformKinematics.position().altitude(), platformKinematics.position().altitude());
		ASSERT_TRUE(newPlatformKinematics.position().timestamp() == propagateTime);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().north(), 600.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().east(), 611.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().down(), 622.);
		ASSERT_TRUE(newPlatformKinematics.acceleration());
		ASSERT_DOUBLE_EQ(newPlatformKinematics.acceleration().value().north(), 50);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.acceleration().value().east(), 51);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.acceleration().value().down(), 52.);
		ASSERT_FALSE(newPlatformKinematics.orientation());
		ASSERT_FALSE(newPlatformKinematics.orientationRate());
		ASSERT_FALSE(newPlatformKinematics.orientationAcceleration());

		platformKinematics = PlatformKinematics{Point4d{0.1, 0.2, 10'000., now}, Velocity3d{-100., -101., -102.}, Acceleration3d{-50., -51., -52.}};
		newPlatformKinematics = PropagatePlatformKinematics(platformKinematics, propagateTime);
		ASSERT_LT(newPlatformKinematics.position().latitude(), platformKinematics.position().latitude());
		ASSERT_LT(newPlatformKinematics.position().longitude(), platformKinematics.position().longitude());
		ASSERT_GT(newPlatformKinematics.position().altitude(), platformKinematics.position().altitude());
		ASSERT_TRUE(newPlatformKinematics.position().timestamp() == propagateTime);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().north(), -600);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().east(), -611.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().down(), -622.);
		ASSERT_TRUE(newPlatformKinematics.acceleration());
		ASSERT_DOUBLE_EQ(newPlatformKinematics.acceleration().value().north(), -50);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.acceleration().value().east(), -51);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.acceleration().value().down(), -52);
		ASSERT_FALSE(newPlatformKinematics.orientation());
		ASSERT_FALSE(newPlatformKinematics.orientationRate());
		ASSERT_FALSE(newPlatformKinematics.orientationAcceleration());

		platformKinematics = PlatformKinematics{Point4d{0.1, 0.2, 10'000., now}, Velocity3d{100., 101., 102.}, Acceleration3d{50, 51, 52.},
												Orientation{0.1, 0.2, 0.3}};
		newPlatformKinematics = PropagatePlatformKinematics(platformKinematics, propagateTime);
		ASSERT_GT(newPlatformKinematics.position().latitude(), platformKinematics.position().latitude());
		ASSERT_GT(newPlatformKinematics.position().longitude(), platformKinematics.position().longitude());
		ASSERT_LT(newPlatformKinematics.position().altitude(), platformKinematics.position().altitude());
		ASSERT_TRUE(newPlatformKinematics.position().timestamp() == propagateTime);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().north(), 600.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().east(), 611.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().down(), 622.);
		ASSERT_TRUE(newPlatformKinematics.acceleration());
		ASSERT_DOUBLE_EQ(newPlatformKinematics.acceleration().value().north(), 50.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.acceleration().value().east(), 51.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.acceleration().value().down(), 52.);
		ASSERT_TRUE(newPlatformKinematics.orientation());
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientation().value().yaw(), 0.1);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientation().value().pitch(), 0.2);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientation().value().roll(), 0.3);
		ASSERT_FALSE(newPlatformKinematics.orientationRate());
		ASSERT_FALSE(newPlatformKinematics.orientationAcceleration());

		platformKinematics = PlatformKinematics{Point4d{0.1, 0.2, 10'000., now}, Velocity3d{100., 101., 102.}, Acceleration3d{50, 51, 52.},
												Orientation{0.1, 0.2, 0.3}, OrientationRate{0.1, 0.2, 0.3}};
		newPlatformKinematics = PropagatePlatformKinematics(platformKinematics, propagateTime);
		ASSERT_GT(newPlatformKinematics.position().latitude(), platformKinematics.position().latitude());
		ASSERT_GT(newPlatformKinematics.position().longitude(), platformKinematics.position().longitude());
		ASSERT_LT(newPlatformKinematics.position().altitude(), platformKinematics.position().altitude());
		ASSERT_TRUE(newPlatformKinematics.position().timestamp() == propagateTime);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().north(), 600.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().east(), 611.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().down(), 622.);
		ASSERT_TRUE(newPlatformKinematics.acceleration());
		ASSERT_DOUBLE_EQ(newPlatformKinematics.acceleration().value().north(), 50);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.acceleration().value().east(), 51);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.acceleration().value().down(), 52.);
		ASSERT_TRUE(newPlatformKinematics.orientation());
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientation().value().yaw(), 1.1);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientation().value().pitch(), 2.2);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientation().value().roll(), 3.3);
		ASSERT_TRUE(newPlatformKinematics.orientationRate());
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientationRate().value().yaw(), 0.1);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientationRate().value().pitch(), 0.2);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientationRate().value().roll(), 0.3);
		ASSERT_FALSE(newPlatformKinematics.orientationAcceleration());

		platformKinematics =
			PlatformKinematics{Point4d{0.1, 0.2, 10'000., now}, Velocity3d{100., 101., 102.},	Acceleration3d{50, 51, 52.},
							   Orientation{0.1, 0.2, 0.3},		OrientationRate{0.1, 0.2, 0.3}, OrientationAcceleration{0.1, 0.2, 0.3}};
		newPlatformKinematics = PropagatePlatformKinematics(platformKinematics, propagateTime);
		ASSERT_GT(newPlatformKinematics.position().latitude(), platformKinematics.position().latitude());
		ASSERT_GT(newPlatformKinematics.position().longitude(), platformKinematics.position().longitude());
		ASSERT_LT(newPlatformKinematics.position().altitude(), platformKinematics.position().altitude());
		ASSERT_TRUE(newPlatformKinematics.position().timestamp() == propagateTime);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().north(), 600.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().east(), 611.);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.velocity().down(), 622.);
		ASSERT_TRUE(newPlatformKinematics.acceleration());
		ASSERT_DOUBLE_EQ(newPlatformKinematics.acceleration().value().north(), 50);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.acceleration().value().east(), 51);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.acceleration().value().down(), 52.);
		ASSERT_TRUE(newPlatformKinematics.orientation());
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientation().value().yaw(), 6.1);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientation().value().pitch(), 12.2);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientation().value().roll(), 18.3);
		ASSERT_TRUE(newPlatformKinematics.orientationRate());
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientationRate().value().yaw(), 1.1);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientationRate().value().pitch(), 2.2);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientationRate().value().roll(), 3.3);
		ASSERT_TRUE(newPlatformKinematics.orientationAcceleration());
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientationAcceleration().value().yaw(), 0.1);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientationAcceleration().value().pitch(), 0.2);
		ASSERT_DOUBLE_EQ(newPlatformKinematics.orientationAcceleration().value().roll(), 0.3);
	}
	/**
	 * @brief test Sqr
	 */
	TEST_F(UtilsTestSuite, TestSqr)
	{
		constexpr int NUM_VALS = 100;
		for(int i = 0; i < NUM_VALS; ++i)
		{
			ASSERT_EQ(Sqr(i), i * i);
		}

		for(int i = 0; i < NUM_VALS; ++i)
		{
			ASSERT_NEAR(Sqr(static_cast<double>(i)), static_cast<double>(i) * static_cast<double>(i), std::numeric_limits<double>::epsilon());
		}
	}

	/**
	 * @brief test transforms
	 */
	TEST_F(UtilsTestSuite, TestTransforms)
	{
		auto lla = reference_frames::LLA{};
		ASSERT_NEAR(lla.latitude(), 0.0, eps);
		ASSERT_NEAR(lla.longitude(), 0.0, eps);
		ASSERT_NEAR(lla.altitude(), 0.0, eps);

		auto ecef = reference_frames::ECEF{};
		ASSERT_NEAR(ecef.x(), 0.0, eps);
		ASSERT_NEAR(ecef.y(), 0.0, eps);
		ASSERT_NEAR(ecef.z(), 0.0, eps);

		auto enu = reference_frames::ENU{};
		ASSERT_NEAR(enu.east(), 0.0, eps);
		ASSERT_NEAR(enu.north(), 0.0, eps);
		ASSERT_NEAR(enu.up(), 0.0, eps);

		auto ned = reference_frames::NED{};
		ASSERT_NEAR(ned.east(), 0.0, eps);
		ASSERT_NEAR(ned.north(), 0.0, eps);
		ASSERT_NEAR(ned.down(), 0.0, eps);

		auto azEl = reference_frames::AzEl{};
		ASSERT_NEAR(azEl.az(), 0.0, eps);
		ASSERT_NEAR(azEl.el(), 0.0, eps);

		constexpr auto transformEps = 1e-8;

		auto nedT = transforms::ENUtoNED(reference_frames::ENU{1.0, 2.0, 3.0});
		ASSERT_NEAR(nedT.north(), 2.0, eps);
		ASSERT_NEAR(nedT.east(), 1.0, eps);
		ASSERT_NEAR(nedT.down(), -3.0, eps);

		auto enuT = transforms::NEDtoENU(reference_frames::NED{1.0, 2.0, 3.0});
		ASSERT_NEAR(enuT.east(), 2.0, eps);
		ASSERT_NEAR(enuT.north(), 1.0, eps);
		ASSERT_NEAR(enuT.up(), -3.0, eps);

		ecef = transforms::LLAtoECEF(reference_frames::LLA{0, 0, 0});
		ASSERT_NEAR(ecef.x(), constants::EARTH_EQUATORIAL_RADIUS, eps);
		ASSERT_NEAR(ecef.y(), 0.0, transformEps);
		ASSERT_NEAR(ecef.z(), 0.0, transformEps);

		ecef = transforms::LLAtoECEF(reference_frames::LLA{0, boost::math::constants::half_pi<double>(), 0});
		ASSERT_NEAR(ecef.x(), 0.0, transformEps);
		ASSERT_NEAR(ecef.y(), constants::EARTH_EQUATORIAL_RADIUS, transformEps);
		ASSERT_NEAR(ecef.z(), 0.0, transformEps);

		ecef = transforms::LLAtoECEF(reference_frames::LLA{0, boost::math::constants::pi<double>(), 0});
		ASSERT_NEAR(ecef.x(), -constants::EARTH_EQUATORIAL_RADIUS, transformEps);
		ASSERT_NEAR(ecef.y(), 0.0, transformEps);
		ASSERT_NEAR(ecef.z(), 0.0, transformEps);

		ecef = transforms::LLAtoECEF(reference_frames::LLA{0, -boost::math::constants::half_pi<double>(), 0});
		ASSERT_NEAR(ecef.x(), 0.0, transformEps);
		ASSERT_NEAR(ecef.y(), -constants::EARTH_EQUATORIAL_RADIUS, transformEps);
		ASSERT_NEAR(ecef.z(), 0.0, transformEps);

		ecef = transforms::LLAtoECEF(reference_frames::LLA{boost::math::constants::half_pi<double>(), 0, 0});
		ASSERT_NEAR(ecef.x(), 0.0, transformEps);
		ASSERT_NEAR(ecef.y(), 0.0, transformEps);
		ASSERT_NEAR(ecef.z(), constants::EARTH_POLAR_RADIUS, transformEps);

		ecef = transforms::LLAtoECEF(reference_frames::LLA{-boost::math::constants::half_pi<double>(), 0, 0});
		ASSERT_NEAR(ecef.x(), 0.0, transformEps);
		ASSERT_NEAR(ecef.y(), 0.0, transformEps);
		ASSERT_NEAR(ecef.z(), -constants::EARTH_POLAR_RADIUS, transformEps);

		ecef = transforms::LLAtoECEF(reference_frames::LLA{0, 0, constants::EARTH_EQUATORIAL_RADIUS});
		ASSERT_NEAR(ecef.x(), 2 * constants::EARTH_EQUATORIAL_RADIUS, transformEps);
		ASSERT_NEAR(ecef.y(), 0.0, transformEps);
		ASSERT_NEAR(ecef.z(), 0.0, transformEps);

		lla = transforms::ECEFtoLLA(reference_frames::ECEF{constants::EARTH_EQUATORIAL_RADIUS, 0, 0});
		ASSERT_NEAR(lla.latitude(), 0.0, transformEps);
		ASSERT_NEAR(lla.longitude(), 0.0, transformEps);
		ASSERT_NEAR(lla.altitude(), 0.0, transformEps);

		lla = transforms::ECEFtoLLA(reference_frames::ECEF{0, constants::EARTH_EQUATORIAL_RADIUS, 0});
		ASSERT_NEAR(lla.latitude(), 0.0, transformEps);
		ASSERT_NEAR(lla.longitude(), boost::math::constants::half_pi<double>(), transformEps);
		ASSERT_NEAR(lla.altitude(), 0.0, transformEps);

		lla = transforms::ECEFtoLLA(reference_frames::ECEF{0, 0, constants::EARTH_POLAR_RADIUS});
		ASSERT_NEAR(lla.latitude(), boost::math::constants::half_pi<double>(), transformEps);
		ASSERT_NEAR(lla.longitude(), 0.0, transformEps);
		ASSERT_NEAR(lla.altitude(), 0.0, transformEps);

		lla = transforms::ECEFtoLLA(reference_frames::ECEF{0, 0, -constants::EARTH_POLAR_RADIUS});
		ASSERT_NEAR(lla.latitude(), -boost::math::constants::half_pi<double>(), transformEps);
		ASSERT_NEAR(lla.longitude(), 0.0, transformEps);
		ASSERT_NEAR(lla.altitude(), 0.0, transformEps);

		lla = transforms::ECEFtoLLA(reference_frames::ECEF{2 * constants::EARTH_EQUATORIAL_RADIUS, 0, 0});
		ASSERT_NEAR(lla.latitude(), 0.0, transformEps);
		ASSERT_NEAR(lla.longitude(), 0.0, transformEps);
		ASSERT_NEAR(lla.altitude(), constants::EARTH_EQUATORIAL_RADIUS, transformEps);

		constexpr double platformAltitude = 100.0;
		const auto platform = reference_frames::LLA{0.0, 0.0, platformAltitude};

		auto target = transforms::LLAtoECEF(reference_frames::LLA{0.0, 0.017453, platformAltitude});
		enu = transforms::ECEFtoENU(target, platform);
		ned = transforms::ECEFtoNED(target, platform);
		ASSERT_GT(enu.east(), 0.0);
		ASSERT_NEAR(enu.north(), 0.0, transformEps);
		ASSERT_LT(enu.up(), 0.0);
		ASSERT_NEAR(enu.east(), ned.east(), transformEps);
		ASSERT_NEAR(enu.north(), ned.north(), transformEps);
		ASSERT_NEAR(enu.up(), -ned.down(), transformEps);

		target = transforms::LLAtoECEF(reference_frames::LLA{0.017453, 0.0, platformAltitude});
		enu = transforms::ECEFtoENU(target, platform);
		ned = transforms::ECEFtoNED(target, platform);
		ASSERT_NEAR(enu.east(), 0.0, transformEps);
		ASSERT_GT(enu.north(), 0.0);
		ASSERT_LT(enu.up(), 0.0);
		ASSERT_NEAR(enu.east(), ned.east(), transformEps);
		ASSERT_NEAR(enu.north(), ned.north(), transformEps);
		ASSERT_NEAR(enu.up(), -ned.down(), transformEps);

		target = transforms::LLAtoECEF(reference_frames::LLA{0.017453, 0.017453, platformAltitude});
		enu = transforms::ECEFtoENU(target, platform);
		ned = transforms::ECEFtoNED(target, platform);
		ASSERT_GT(enu.east(), 0.0);
		ASSERT_GT(enu.north(), 0.0);
		ASSERT_LT(enu.up(), 0.0);
		ASSERT_NEAR(enu.east(), ned.east(), transformEps);
		ASSERT_NEAR(enu.north(), ned.north(), transformEps);
		ASSERT_NEAR(enu.up(), -ned.down(), transformEps);

		target = transforms::LLAtoECEF(reference_frames::LLA{0.017453, 0.017453, platformAltitude + 10000});
		enu = transforms::ECEFtoENU(target, platform);
		ned = transforms::ECEFtoNED(target, platform);
		ASSERT_GT(enu.east(), 0.0);
		ASSERT_GT(enu.north(), 0.0);
		ASSERT_GT(enu.up(), 0.0);
		ASSERT_NEAR(enu.east(), ned.east(), transformEps);
		ASSERT_NEAR(enu.north(), ned.north(), transformEps);
		ASSERT_NEAR(enu.up(), -ned.down(), transformEps);

		auto targetLLA = reference_frames::LLA{0.017453, 0.0, platformAltitude};
		azEl = transforms::LLAtoAzElGeocentric(targetLLA, platform);
		ASSERT_NEAR(azEl.az(), 0.0, transformEps);
		ASSERT_LT(azEl.el(), 0.0);

		targetLLA = reference_frames::LLA{0.0, 0.017453, platformAltitude};
		azEl = transforms::LLAtoAzElGeocentric(targetLLA, platform);
		ASSERT_NEAR(azEl.az(), boost::math::constants::half_pi<double>(), transformEps);
		ASSERT_LT(azEl.el(), 0.0);

		targetLLA = reference_frames::LLA{-0.017453, 0.0, platformAltitude};
		azEl = transforms::LLAtoAzElGeocentric(targetLLA, platform);
		ASSERT_NEAR(azEl.az(), boost::math::constants::pi<double>(), transformEps);
		ASSERT_LT(azEl.el(), 0.0);

		targetLLA = reference_frames::LLA{0.0, -0.017453, platformAltitude};
		azEl = transforms::LLAtoAzElGeocentric(targetLLA, platform);
		ASSERT_NEAR(azEl.az(), boost::math::constants::half_pi<double>() + boost::math::constants::pi<double>(), transformEps);
		ASSERT_LT(azEl.el(), 0.0);

		targetLLA = reference_frames::LLA{0.017453, 0.0, platformAltitude + 10000};
		azEl = transforms::LLAtoAzElGeocentric(targetLLA, platform);
		ASSERT_NEAR(azEl.az(), 0.0, transformEps);
		ASSERT_GT(azEl.el(), 0.0);

		constexpr int NUM_POINTS = 100;

		auto rng = std::mt19937_64{};
		rng.seed(666);
		std::uniform_real_distribution<double> distLat(-boost::math::constants::half_pi<double>(), boost::math::constants::half_pi<double>());
		std::uniform_real_distribution<double> distLon(-boost::math::constants::pi<double>(), boost::math::constants::pi<double>());
		std::uniform_real_distribution<double> distAlt(100, 10000);

		for(int i = 0; i < NUM_POINTS; ++i)
		{
			const auto platformPosition = reference_frames::LLA{distLat(rng), distLon(rng), distAlt(rng)};
			const auto targetPosition = reference_frames::LLA{distLat(rng), distLon(rng), distAlt(rng)};

			const auto azEl1 = transforms::LLAtoAzElGeocentric(targetPosition, platformPosition);
			const auto azEl2 = transforms::LLAtoAzElGeodetic(targetPosition, platformPosition);
			ASSERT_NEAR(azEl1.az(), azEl2.az(), 0.1);
			ASSERT_NEAR(azEl1.el(), azEl2.el(), 0.1);
		}

		// AFSIM outputs
		const auto platform2ECEF = reference_frames::ECEF{889780.8040509718, -5443884.478448521, 3191301.5726495585};
		const auto platform2Euler = Euler{1.678885817527771, -1.0427558422088623, -3.0950019359588623};
		const auto platform2RollPitchYaw = Orientation(0.027159271086905079, 0., 0.);

		const auto platform2RollPitchYawCalc = transforms::ECEFEulerToNEDRollPitchYaw(platform2ECEF, platform2Euler);
		ASSERT_NEAR(platform2RollPitchYawCalc.roll(), platform2RollPitchYaw.roll(), 1e-7);
		ASSERT_NEAR(platform2RollPitchYawCalc.pitch(), platform2RollPitchYaw.pitch(), 1e-7);
		ASSERT_NEAR(platform2RollPitchYawCalc.yaw(), platform2RollPitchYaw.yaw(), 1e-7);

		const auto platform2EulerCalc = transforms::NEDRollPitchYawToECEFEuler(platform2ECEF, platform2RollPitchYaw);
		ASSERT_NEAR(platform2EulerCalc.psi(), platform2Euler.psi(), 1e-7);
		ASSERT_NEAR(platform2EulerCalc.theta(), platform2Euler.theta(), 1e-7);
		ASSERT_NEAR(platform2EulerCalc.phi(), platform2Euler.phi(), 1e-7);

		const auto platform3ECEF = reference_frames::ECEF{-1288345.7521444533, -4718928.642526492, 4079259.935028878};
		const auto platform3Euler = Euler{1.30427503581543, -.872403085231781, 3.1415927410125732};
		const auto platform3RollPitchYaw = Orientation(0., 0., 0.);

		const auto platform3RollPitchYawCalc = transforms::ECEFEulerToNEDRollPitchYaw(platform3ECEF, platform3Euler);
		ASSERT_NEAR(platform3RollPitchYawCalc.roll(), platform3RollPitchYaw.roll(), 1e-7);
		ASSERT_NEAR(platform3RollPitchYawCalc.pitch(), platform3RollPitchYaw.pitch(), 1e-7);
		ASSERT_NEAR(platform3RollPitchYawCalc.yaw(), platform3RollPitchYaw.yaw(), 1e-7);

		const auto platform3EulerCalc = transforms::NEDRollPitchYawToECEFEuler(platform3ECEF, platform3RollPitchYaw);
		ASSERT_NEAR(platform3EulerCalc.psi(), platform3Euler.psi(), 1e-7);
		ASSERT_NEAR(platform3EulerCalc.theta(), platform3Euler.theta(), 1e-7);
		ASSERT_NEAR(platform3EulerCalc.phi(), -platform3Euler.phi(), 1e-7);

		const auto platform4ECEF = reference_frames::ECEF{861284.8918511268, -5441200.936501232, 3203589.383938122};
		const auto platform4Euler = Euler{-2.4969322681427, -0.4192129075527191, 2.2737600803375244};
		const auto platform4RollPitchYaw = Orientation(-1.4049900478554354, 0.6126105674500097, 0.33161255787892263);

		const auto platform4RollPitchYawCalc = transforms::ECEFEulerToNEDRollPitchYaw(platform4ECEF, platform4Euler);
		ASSERT_NEAR(platform4RollPitchYawCalc.roll(), platform4RollPitchYaw.roll(), 1e-7);
		ASSERT_NEAR(platform4RollPitchYawCalc.pitch(), platform4RollPitchYaw.pitch(), 1e-7);
		ASSERT_NEAR(platform4RollPitchYawCalc.yaw(), platform4RollPitchYaw.yaw(), 1e-7);

		const auto platform4EulerCalc = transforms::NEDRollPitchYawToECEFEuler(platform4ECEF, platform4RollPitchYaw);
		ASSERT_NEAR(platform4EulerCalc.psi(), platform4Euler.psi(), 1e-7);
		ASSERT_NEAR(platform4EulerCalc.theta(), platform4Euler.theta(), 1e-7);
		ASSERT_NEAR(platform4EulerCalc.phi(), platform4Euler.phi(), 1e-7);
	}

	/**
	 * @brief test UUIDfromStr
	 */
	TEST_F(UtilsTestSuite, TestECEFtoBody)
	{
		const auto platformECEF = transforms::LLAtoECEF(reference_frames::LLA{0., 0., 0.});
		auto platformOrientation = types::Orientation{0., 0., 0.};

		auto bodyAzEl = transforms::ECEFtoBody(platformECEF, platformOrientation, transforms::LLAtoECEF(reference_frames::LLA{0.01, 0., 0.}));
		ASSERT_NEAR(bodyAzEl.az().position(), 0., 1e-7);
		ASSERT_NEAR(bodyAzEl.el().position(), 0., 0.1);

		bodyAzEl = transforms::ECEFtoBody(platformECEF, platformOrientation, transforms::LLAtoECEF(reference_frames::LLA{-0.01, 0., 0.}));
		ASSERT_NEAR(bodyAzEl.az().position(), boost::math::constants::pi<double>(), 1e-7);
		ASSERT_NEAR(bodyAzEl.el().position(), 0., 0.1);

		bodyAzEl = transforms::ECEFtoBody(platformECEF, platformOrientation, transforms::LLAtoECEF(reference_frames::LLA{0., 0.1, 0.}));
		ASSERT_NEAR(bodyAzEl.az().position(), boost::math::constants::half_pi<double>(), 1e-7);
		ASSERT_NEAR(bodyAzEl.el().position(), 0., 0.1);

		bodyAzEl = transforms::ECEFtoBody(platformECEF, platformOrientation, transforms::LLAtoECEF(reference_frames::LLA{0., -0.1, 0.}));
		ASSERT_NEAR(bodyAzEl.az().position(), boost::math::constants::pi<double>() + boost::math::constants::half_pi<double>(), 1e-7);
		ASSERT_NEAR(bodyAzEl.el().position(), 0., 0.1);

		platformOrientation = types::Orientation{boost::math::constants::half_pi<double>(), 0., 0.};
		bodyAzEl = transforms::ECEFtoBody(platformECEF, platformOrientation, transforms::LLAtoECEF(reference_frames::LLA{0.01, 0., 0.}));
		ASSERT_NEAR(bodyAzEl.az().position(), boost::math::constants::pi<double>() + boost::math::constants::half_pi<double>(), 1e-7);
		ASSERT_NEAR(bodyAzEl.el().position(), 0., 0.1);

		platformOrientation = types::Orientation{-boost::math::constants::half_pi<double>(), 0., 0.};
		bodyAzEl = transforms::ECEFtoBody(platformECEF, platformOrientation, transforms::LLAtoECEF(reference_frames::LLA{0.01, 0., 0.}));
		ASSERT_NEAR(bodyAzEl.az().position(), boost::math::constants::half_pi<double>(), 1e-7);
		ASSERT_NEAR(bodyAzEl.el().position(), 0., 0.1);

		platformOrientation = types::Orientation{0., boost::math::constants::half_pi<double>() / 2., 0.};
		bodyAzEl = transforms::ECEFtoBody(platformECEF, platformOrientation, transforms::LLAtoECEF(reference_frames::LLA{0.01, 0., 0.}));
		ASSERT_NEAR(bodyAzEl.az().position(), 0., 1e-7);
		ASSERT_NEAR(bodyAzEl.el().position(), -boost::math::constants::half_pi<double>() / 2., 0.1);

		platformOrientation = types::Orientation{0., -boost::math::constants::half_pi<double>() / 2., 0.};
		bodyAzEl = transforms::ECEFtoBody(platformECEF, platformOrientation, transforms::LLAtoECEF(reference_frames::LLA{0.01, 0., 0.}));
		ASSERT_NEAR(bodyAzEl.az().position(), 0., 1e-7);
		ASSERT_NEAR(bodyAzEl.el().position(), boost::math::constants::half_pi<double>() / 2., 0.1);

		platformOrientation = types::Orientation{0., 0., boost::math::constants::half_pi<double>() / 2.};
		bodyAzEl = transforms::ECEFtoBody(platformECEF, platformOrientation, transforms::LLAtoECEF(reference_frames::LLA{0., 0.01, 0.}));
		ASSERT_NEAR(bodyAzEl.az().position(), boost::math::constants::half_pi<double>(), 1e-7);
		ASSERT_NEAR(bodyAzEl.el().position(), boost::math::constants::half_pi<double>() / 2., 0.1);

		platformOrientation = types::Orientation{0., 0., -boost::math::constants::half_pi<double>() / 2.};
		bodyAzEl = transforms::ECEFtoBody(platformECEF, platformOrientation, transforms::LLAtoECEF(reference_frames::LLA{0., 0.01, 0.}));
		ASSERT_NEAR(bodyAzEl.az().position(), boost::math::constants::half_pi<double>(), 1e-7);
		ASSERT_NEAR(bodyAzEl.el().position(), -boost::math::constants::half_pi<double>() / 2., 0.1);
	}

	/**
	 * @brief test UUIDfromStr
	 */
	TEST_F(UtilsTestSuite, TestUuidFromStr)
	{
		const auto uuid = GenerateUUID();
		ASSERT_EQ(UUIDfromStr(UUIDtoStr(uuid)), uuid);
	}

	/**
	 * @brief test UUIDtoStr
	 */
	TEST_F(UtilsTestSuite, TestUuidToStr)
	{
		ASSERT_NO_THROW([[maybe_unused]] const auto idStr = UUIDtoStr(GenerateUUID()));
	}

	/**
	 * @brief test Wrap
	 */
	TEST_F(UtilsTestSuite, TestWrap)
	{
		constexpr int NUM_VALS = 100;

		ASSERT_NEAR(Wrap(0.0), 0.0, std::numeric_limits<double>::epsilon());
		auto angle1 = boost::math::constants::pi<double>() - 0.01;
		ASSERT_NEAR(Wrap(angle1), angle1, std::numeric_limits<double>::epsilon());
		auto angle2 = boost::math::constants::pi<double>() + 0.01;
		ASSERT_NEAR(Wrap(angle2), -angle1, std::numeric_limits<double>::epsilon());

		auto rng = std::mt19937_64{};
		rng.seed(666);
		std::uniform_real_distribution<double> dist(0.0, boost::math::constants::two_pi<double>());

		for(int i = 0; i < NUM_VALS; ++i)
		{
			const auto value = Wrap(dist(rng));
			ASSERT_LE(value, boost::math::constants::pi<double>());
			ASSERT_GE(value, -boost::math::constants::pi<double>());
		}
	}

	/**
	 * @brief test Wrap2PI
	 */
	TEST_F(UtilsTestSuite, TestWrap2Pi)
	{
		constexpr int NUM_VALS = 100;

		ASSERT_NEAR(Wrap2Pi(0.0), 0.0, std::numeric_limits<double>::epsilon());
		auto angle1 = boost::math::constants::two_pi<double>() - 0.01;
		ASSERT_NEAR(Wrap2Pi(angle1), angle1, std::numeric_limits<double>::epsilon());
		auto angle2 = boost::math::constants::two_pi<double>() + 0.01;
		ASSERT_NEAR(Wrap2Pi(angle2), 0.01, std::numeric_limits<double>::epsilon());

		auto rng = std::mt19937_64{};
		rng.seed(666);
		std::uniform_real_distribution<double> dist(0.0, 2 * boost::math::constants::two_pi<double>());

		for(int i = 0; i < NUM_VALS; ++i)
		{
			const auto value = Wrap2Pi(dist(rng));
			ASSERT_LE(value, boost::math::constants::two_pi<double>());
			ASSERT_GE(value, 0);
		}
	}

	/**
	 * @brief test CartToInert
	 */
	TEST_F(UtilsTestSuite, TestCartToInertial)
	{
		auto cart = Cartesian(1.0, 0.0, 0.0);
		auto point = transforms::CartToInertial(cart);
		ASSERT_NEAR(point.az().position(), 0.0, std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(point.el().position(), 0.0, std::numeric_limits<double>::epsilon());

		cart = Cartesian(0.0, 1.0, 0.0);
		point = transforms::CartToInertial(cart);
		ASSERT_NEAR(point.az().position(), boost::math::constants::half_pi<double>(), std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(point.el().position(), 0.0, std::numeric_limits<double>::epsilon());

		cart = Cartesian(-1.0, 0.0, 0.0);
		point = transforms::CartToInertial(cart);
		ASSERT_NEAR(point.az().position(), boost::math::constants::pi<double>(), std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(point.el().position(), 0.0, std::numeric_limits<double>::epsilon());

		cart = Cartesian(0.0, -1.0, 0.0);
		point = transforms::CartToInertial(cart);
		ASSERT_NEAR(point.az().position(), boost::math::constants::pi<double>() + boost::math::constants::half_pi<double>(),
					std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(point.el().position(), 0.0, std::numeric_limits<double>::epsilon());

		cart = Cartesian(0.0, 0.0, 1.0);
		point = transforms::CartToInertial(cart);
		ASSERT_NEAR(point.az().position(), 0.0, std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(point.el().position(), -boost::math::constants::half_pi<double>(), std::numeric_limits<double>::epsilon());

		cart = Cartesian(0.0, 0.0, -1.0);
		point = transforms::CartToInertial(cart);
		ASSERT_NEAR(point.az().position(), 0.0, std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(point.el().position(), boost::math::constants::half_pi<double>(), std::numeric_limits<double>::epsilon());
	}

	/**
	 * @brief test Inertial to Cartesian coordinates
	 */
	TEST_F(UtilsTestSuite, TestInertialToCart)
	{
		auto point = Point2dInertial(InertialAxis(0.0), InertialAxis(0.0));
		auto cart = transforms::InertialToCart(point);
		ASSERT_NEAR(cart.x(), 1.0, std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(cart.y(), 0.0, std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(cart.z(), 0.0, std::numeric_limits<double>::epsilon());

		point = Point2dInertial(InertialAxis(boost::math::constants::half_pi<double>()), InertialAxis(0.0));
		cart = transforms::InertialToCart(point);
		ASSERT_NEAR(cart.x(), 0.0, std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(cart.y(), 1.0, std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(cart.z(), 0.0, std::numeric_limits<double>::epsilon());

		point = Point2dInertial(InertialAxis(boost::math::constants::pi<double>()), InertialAxis(0.0));
		cart = transforms::InertialToCart(point);
		ASSERT_NEAR(cart.x(), -1.0, std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(cart.y(), 0.0, std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(cart.z(), 0.0, std::numeric_limits<double>::epsilon());

		point = Point2dInertial(InertialAxis(boost::math::constants::pi<double>() + boost::math::constants::half_pi<double>()), InertialAxis(0.0));
		cart = transforms::InertialToCart(point);
		ASSERT_NEAR(cart.x(), 0.0, std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(cart.y(), -1.0, std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(cart.z(), 0.0, std::numeric_limits<double>::epsilon());

		point = Point2dInertial(InertialAxis(0.0), InertialAxis(-boost::math::constants::half_pi<double>()));
		cart = transforms::InertialToCart(point);
		ASSERT_NEAR(cart.x(), 0.0, std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(cart.y(), 0.0, std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(cart.z(), 1.0, std::numeric_limits<double>::epsilon());

		point = Point2dInertial(InertialAxis(0.0), InertialAxis(boost::math::constants::half_pi<double>()));
		cart = transforms::InertialToCart(point);
		ASSERT_NEAR(cart.x(), 0.0, std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(cart.y(), 0.0, std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(cart.z(), -1.0, std::numeric_limits<double>::epsilon());

		point = Point2dInertial(InertialAxis(boost::math::constants::half_pi<double>() / 2.),
								InertialAxis(boost::math::constants::half_pi<double>() / 2.));
		cart = transforms::InertialToCart(point);
		ASSERT_NEAR(cart.x(), 0.5, std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(cart.y(), 0.5, std::numeric_limits<double>::epsilon());
		ASSERT_NEAR(cart.z(), -0.7071067811865476, std::numeric_limits<double>::epsilon());
	}
} // namespace ball::common::utils::math
  // UNCLASSIFIED
