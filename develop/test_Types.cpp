// UNCLASSIFIED

#include "gtest/gtest.h"
#include "test.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <variant>

#include "ball/common/utils/Math.h"

using namespace ball::common::utils::math::types;
using namespace ball::common::utils::math::utils;

namespace ball::common::utils::math
{
	class TypesTestSuite : public BaseTest
	{
	};

	/**
	 * @brief test Acceleration 3d
	 */
	TEST_F(TypesTestSuite, TestAcceleration3d)
	{
		auto a = Acceleration3d();
		ASSERT_EQ(a.north(), 0.0);
		ASSERT_EQ(a.east(), 0.0);
		ASSERT_EQ(a.down(), 0.0);
		ASSERT_FALSE(a.timestamp());

		double north = 1.0;
		double east = 2.0;
		double down = 3.0;
		a = Acceleration3d(north, east, down);
		ASSERT_EQ(a.north(), north);
		ASSERT_EQ(a.east(), east);
		ASSERT_EQ(a.down(), down);
		ASSERT_FALSE(a.timestamp());

		auto timestamp = Clock::now();
		a = Acceleration3d(north, east, down, timestamp);
		ASSERT_EQ(a.north(), north);
		ASSERT_EQ(a.east(), east);
		ASSERT_EQ(a.down(), down);
		ASSERT_TRUE(a.timestamp());

		a = Acceleration3d();
		a.setNorth(north);
		a.setEast(east);
		a.setDown(down);
		a.setTimestamp(timestamp);
		ASSERT_EQ(a.north(), north);
		ASSERT_EQ(a.east(), east);
		ASSERT_EQ(a.down(), down);
		ASSERT_TRUE(a.timestamp());

		auto a2 = a;
		ASSERT_TRUE(a == a2);
		ASSERT_FALSE(a != a2);
	}

	/**
	 * @brief test AirVolumeSensorReferenced
	 */
	TEST_F(TypesTestSuite, TestAirVolumeSensorReferenced)
	{
		auto a = AirVolumeSensorReferenced();
		ASSERT_EQ(a.azScanWidth(), 0.0);
		ASSERT_EQ(a.elScanWidth(), 0.0);
		ASSERT_EQ(a.azScanCenter(), 0.0);
		ASSERT_EQ(a.elScanCenter(), 0.0);
		ASSERT_EQ(a.minRangeOfInterest(), 0.0);
		ASSERT_EQ(a.maxRangeOfInterest(), std::numeric_limits<double>::max());

		double azScanWidth = 1.0;
		double elScanWidth = 2.0;
		double azScanCenter = 3.0;
		double elScanCenter = 4.0;
		a = AirVolumeSensorReferenced(azScanWidth, elScanWidth, azScanCenter, elScanCenter);
		ASSERT_EQ(a.azScanWidth(), azScanWidth);
		ASSERT_EQ(a.elScanWidth(), elScanWidth);
		ASSERT_EQ(a.azScanCenter(), azScanCenter);
		ASSERT_EQ(a.elScanCenter(), elScanCenter);
		ASSERT_EQ(a.minRangeOfInterest(), 0.0);
		ASSERT_EQ(a.maxRangeOfInterest(), std::numeric_limits<double>::max());

		double minRangeOfInterest = 5.0;
		double maxRangeOfInterest = 6.0;
		a = AirVolumeSensorReferenced(azScanWidth, elScanWidth, azScanCenter, elScanCenter, minRangeOfInterest, maxRangeOfInterest);
		ASSERT_EQ(a.azScanWidth(), azScanWidth);
		ASSERT_EQ(a.elScanWidth(), elScanWidth);
		ASSERT_EQ(a.azScanCenter(), azScanCenter);
		ASSERT_EQ(a.elScanCenter(), elScanCenter);
		ASSERT_EQ(a.minRangeOfInterest(), minRangeOfInterest);
		ASSERT_EQ(a.maxRangeOfInterest(), maxRangeOfInterest);

		a = AirVolumeSensorReferenced();
		a.setAzScanWidth(azScanWidth);
		a.setElScanWidth(elScanWidth);
		a.setAzScanCenter(azScanCenter);
		a.setElScanCenter(elScanCenter);
		a.setMinRangeOfInterest(minRangeOfInterest);
		a.setMaxRangeOfInterest(maxRangeOfInterest);
		ASSERT_EQ(a.azScanWidth(), azScanWidth);
		ASSERT_EQ(a.elScanWidth(), elScanWidth);
		ASSERT_EQ(a.azScanCenter(), azScanCenter);
		ASSERT_EQ(a.elScanCenter(), elScanCenter);
		ASSERT_EQ(a.minRangeOfInterest(), minRangeOfInterest);
		ASSERT_EQ(a.maxRangeOfInterest(), maxRangeOfInterest);

		auto a2 = a;
		ASSERT_TRUE(a == a2);
		ASSERT_FALSE(a != a2);
	}

	/**
	 * @brief test AltitudeRange
	 */
	TEST_F(TypesTestSuite, TestAltitudeRange)
	{
		auto a = AltitudeRange();
		ASSERT_EQ(a.min(), 0.0);
		ASSERT_EQ(a.max(), 0.0);

		double min = 1.0;
		double max = 2.0;
		a = AltitudeRange(min, max);

		auto a2 = a;
		ASSERT_TRUE(a == a2);
		ASSERT_FALSE(a != a2);
	}

	/**
	 * @brief test Clock
	 */
	TEST_F(TypesTestSuite, TestClock)
	{
		auto now1 = Clock::now();
		std::cout << now1 << '\n';

		auto now2 = Clock::now();
		auto diff = now2 - now1;
		std::cout << diff << '\n';
	}

	/**
	 * @brief test DateTime
	 */
	TEST_F(TypesTestSuite, TestDateTime)
	{
		auto d = DateTime();
		ASSERT_EQ(d.year(), 1970);
		ASSERT_EQ(d.month(), 1);
		ASSERT_EQ(d.day(), 1);
		ASSERT_EQ(d.hour(), 0);
		ASSERT_EQ(d.minute(), 0);
		ASSERT_EQ(d.second(), 0);
		ASSERT_EQ(d.fractionalSecond(), 0.0);
		ASSERT_EQ(d.toStr(), "1970-01-01T00:00:00Z");

		const int year = 2000;
		const int month = 2;
		const int day = 3;
		const int hour = 4;
		const int minute = 5;
		const int second = 6;
		d = DateTime(year, month, day, hour, minute, second);
		ASSERT_EQ(d.year(), year);
		ASSERT_EQ(d.month(), month);
		ASSERT_EQ(d.day(), day);
		ASSERT_EQ(d.hour(), hour);
		ASSERT_EQ(d.minute(), minute);
		ASSERT_EQ(d.second(), second);
		ASSERT_EQ(d.fractionalSecond(), 0.0);
		ASSERT_EQ(d.toStr(), "2000-02-03T04:05:06Z");

		const double fractionalSecond = 0.7;
		d = DateTime(year, month, day, hour, minute, second, fractionalSecond);
		ASSERT_EQ(d.year(), year);
		ASSERT_EQ(d.month(), month);
		ASSERT_EQ(d.day(), day);
		ASSERT_EQ(d.hour(), hour);
		ASSERT_EQ(d.minute(), minute);
		ASSERT_EQ(d.second(), second);
		ASSERT_EQ(d.fractionalSecond(), fractionalSecond);
		ASSERT_EQ(d.toStr(), "2000-02-03T04:05:06.7Z");

		d = DateTime();
		d.setYear(year);
		d.setMonth(month);
		d.setDay(day);
		d.setHour(hour);
		d.setMinute(minute);
		d.setSecond(second);
		d.setFractionalSecond(fractionalSecond);
		ASSERT_EQ(d.year(), year);
		ASSERT_EQ(d.month(), month);
		ASSERT_EQ(d.day(), day);
		ASSERT_EQ(d.hour(), hour);
		ASSERT_EQ(d.minute(), minute);
		ASSERT_EQ(d.second(), second);
		ASSERT_EQ(d.fractionalSecond(), fractionalSecond);
		ASSERT_EQ(d.toStr(), "2000-02-03T04:05:06.7Z");

		d = DateTime::now();
		const auto tp = d.toTimePoint();
		ASSERT_GT(tp.time_since_epoch().count(), 0);

		const auto d2 = DateTime::now();

		ASSERT_FALSE(d == d2);
		ASSERT_TRUE(d != d2);
		ASSERT_TRUE(d < d2);
		ASSERT_TRUE(d <= d2);
		ASSERT_FALSE(d > d2);
		ASSERT_FALSE(d >= d2);
		ASSERT_TRUE(d2 - d > Duration{0});

		ASSERT_THROW(DateTime("1-00-29T11:52:33.123456789Z"), std::invalid_argument);
		ASSERT_THROW(DateTime("22-00-29T11:52:33.123456789Z"), std::invalid_argument);
		ASSERT_THROW(DateTime("222-00-29T11:52:33.123456789Z"), std::invalid_argument);
		ASSERT_THROW(DateTime("2022-0-29T11:52:33.123456789Z"), std::invalid_argument);
		ASSERT_THROW(DateTime("2022-00-29T11:52:33.123456789Z"), std::invalid_argument);
		ASSERT_THROW(DateTime("2022-13-29T11:52:33.123456789Z"), std::invalid_argument);
		ASSERT_THROW(DateTime("2022-11-0T11:52:33.123456789Z"), std::invalid_argument);
		ASSERT_THROW(DateTime("2022-11-00T11:52:33.123456789Z"), std::invalid_argument);
		ASSERT_THROW(DateTime("2022-11-32T11:52:33.123456789Z"), std::invalid_argument);
		ASSERT_NO_THROW(DateTime("2022-11-29T24:52:33.123456789Z")); // boost will roll-over
		ASSERT_NO_THROW(DateTime("2022-11-29T11:60:33.123456789Z")); // boost will roll-over
		ASSERT_NO_THROW(DateTime("2022-11-29T11:52:60.123456789Z")); // boost will roll-over
		ASSERT_THROW(DateTime("2022-11-29T11:52:33.123456789"), std::invalid_argument);

		const auto timestamp = std::string{"2022-11-29T11:52:33.123456Z"};
		d = DateTime(timestamp);
		ASSERT_EQ(d.year(), 2022);
		ASSERT_EQ(d.month(), 11);
		ASSERT_EQ(d.day(), 29);
		ASSERT_EQ(d.hour(), 11);
		ASSERT_EQ(d.minute(), 52);
		ASSERT_EQ(d.second(), 33);
		ASSERT_EQ(d.fractionalSecond(), 0.123456);
		ASSERT_EQ(d.toStr(), timestamp);

		// this is pretty cool
		const auto timestamp2 = std::string{"2022-11-29T24:52:33.123456Z"};
		d = DateTime(timestamp2);
		ASSERT_EQ(d.year(), 2022);
		ASSERT_EQ(d.month(), 11);
		ASSERT_EQ(d.day(), 30);
		ASSERT_EQ(d.hour(), 0);
		ASSERT_EQ(d.minute(), 52);
		ASSERT_EQ(d.second(), 33);
		ASSERT_EQ(d.fractionalSecond(), 0.123456);
		ASSERT_EQ(d.toStr(), "2022-11-30T00:52:33.123456Z");

		const auto tp2 = Clock::now();
		d = DateTime(tp2);
		ASSERT_EQ(d.toTimePoint().time_since_epoch().count(), std::chrono::time_point_cast<TimePoint::duration>(tp2).time_since_epoch().count());

		// Edge case
		const auto tp3 = TimePoint{Duration{1}};
		d = DateTime(tp3);
		ASSERT_EQ(d.toStr(), "1970-01-01T00:00:00.000000001Z");
	}

	/**
	 * @brief test EntityDataFromEnvSim and EntityDataToMfa
	 */
	TEST_F(TypesTestSuite, TestEntityData)
	{
		auto e = EntityDataFromEnvSim();
		auto timestamp = Clock::now();
		auto id = EntityId{1, 2, 3};
		auto pos = utils::Cartesian(4, 5, 6);
		auto vel = utils::Cartesian(7, 8, 9);
		auto orienation = Orientation(10, 11, 12);
		double radiance = 13;
		auto res = EntityDataFromEnvSim::ReservedData();
		for(size_t idx = 0; idx < res.size(); idx++)
		{
			res[idx] = idx;
		}

		e.setTimestamp(timestamp);
		e.setId(id);
		e.setPosEcef(pos);
		e.setVelEcef(vel);
		e.setEuler(orienation);
		e.setRadiance(radiance);
		e.setReserved(res);

		ASSERT_TRUE(e.getTimestamp() == timestamp);
		ASSERT_EQ(e.getId(), id);
		ASSERT_EQ(e.getPosEcef(), pos);
		ASSERT_EQ(e.getVelEcef(), vel);
		ASSERT_EQ(e.getEuler(), orienation);
		ASSERT_EQ(e.getRadiance(), radiance);
		ASSERT_EQ(e.getReserved(), res);

		auto e2 = EntityDataToMfa();
		auto updateTime = Clock::now();
		double range = 14;
		double rangeRate = 15;
		auto los = LOS_Option(16, 17);
		bool flag = true;

		e2.setUpdateTime(updateTime);
		e2.setRange(range);
		e2.setRangeRate(rangeRate);
		e2.setPosAzEl(los);
		e2.setSsaTrackFlag(flag);

		ASSERT_TRUE(e2.getUpdateTime() == updateTime);
		ASSERT_EQ(e2.getRange(), range);
		ASSERT_EQ(e2.getRangeRate(), rangeRate);
		ASSERT_EQ(e2.getPosAzEl(), los);
		ASSERT_EQ(e2.getSsaTrackFlag(), flag);
	}

	/**
	 * @brief test Euler
	 */
	TEST_F(TypesTestSuite, TestEuler)
	{
		auto e = Euler();
		ASSERT_EQ(e.psi(), 0.0);
		ASSERT_EQ(e.theta(), 0.0);
		ASSERT_EQ(e.phi(), 0.0);
		ASSERT_FALSE(e.timestamp());

		double psi = 1.0;
		double theta = 2.0;
		double phi = 3.0;
		auto timestamp = Clock::now();

		e = Euler(psi, theta, phi);
		ASSERT_EQ(e.psi(), psi);
		ASSERT_EQ(e.theta(), theta);
		ASSERT_EQ(e.phi(), phi);
		ASSERT_FALSE(e.timestamp());

		e = Euler(psi, theta, phi, timestamp);
		ASSERT_EQ(e.psi(), psi);
		ASSERT_EQ(e.theta(), theta);
		ASSERT_EQ(e.phi(), phi);
		ASSERT_TRUE(e.timestamp().value() == timestamp);

		e = Euler();
		e.setPsi(psi);
		e.setTheta(theta);
		e.setPhi(phi);
		e.setTimestamp(timestamp);
		ASSERT_EQ(e.psi(), psi);
		ASSERT_EQ(e.theta(), theta);
		ASSERT_EQ(e.phi(), phi);
		ASSERT_TRUE(e.timestamp().value() == timestamp);

		auto o2 = e;
		ASSERT_TRUE(e == o2);
		ASSERT_FALSE(e != o2);
	}

	/**
	 * @brief test LineOfBearingAndUncertainty
	 */
	TEST_F(TypesTestSuite, TestLineOfBearingAndUncertainty)
	{
		auto l = LineOfBearingAndUncertainty();
		ASSERT_EQ(l.los(), LOS{});
		ASSERT_FALSE(l.losUncertainty());

		auto los = LOS(1.0, 2.0, 3.0);
		auto losUncertainty = LOSUncertainty(4.0, 5.0, 6.0);

		l = LineOfBearingAndUncertainty(los);
		ASSERT_EQ(l.los(), los);
		ASSERT_FALSE(l.losUncertainty());

		l = LineOfBearingAndUncertainty(los, losUncertainty);
		ASSERT_EQ(l.los(), los);
		ASSERT_EQ(l.losUncertainty().value(), losUncertainty);

		l = LineOfBearingAndUncertainty();
		l.setLos(los);
		l.setLosUncertainty(losUncertainty);
		ASSERT_EQ(l.los(), los);
		ASSERT_EQ(l.losUncertainty().value(), losUncertainty);

		auto l2 = l;
		ASSERT_TRUE(l == l2);
		ASSERT_FALSE(l != l2);
	}

	/**
	 * @brief test LOS_Option
	 */
	TEST_F(TypesTestSuite, TestLOS_Option)
	{
		auto l = LOS_Option();
		ASSERT_EQ(l.az(), 0.0);
		ASSERT_EQ(l.el(), 0.0);
		ASSERT_FALSE(l.roll());
		ASSERT_FALSE(l.azRate());
		ASSERT_FALSE(l.elRate());
		ASSERT_FALSE(l.rollRate());

		double az = 1.0;
		double el = 2.0;
		double roll = 3.0;
		double azRate = 4.0;
		double elRate = 5.0;
		double rollRate = 6.0;

		l = LOS_Option(az, el);
		ASSERT_EQ(l.az(), az);
		ASSERT_EQ(l.el(), el);
		ASSERT_FALSE(l.roll());
		ASSERT_FALSE(l.azRate());
		ASSERT_FALSE(l.elRate());
		ASSERT_FALSE(l.rollRate());

		l = LOS_Option(az, azRate, el, elRate);
		ASSERT_EQ(l.az(), az);
		ASSERT_EQ(l.el(), el);
		ASSERT_FALSE(l.roll());
		ASSERT_EQ(l.azRate().value(), azRate);
		ASSERT_EQ(l.elRate().value(), elRate);
		ASSERT_FALSE(l.rollRate());

		l = LOS_Option(az, el, roll);
		ASSERT_EQ(l.az(), az);
		ASSERT_EQ(l.el(), el);
		ASSERT_EQ(l.roll().value(), roll);
		ASSERT_FALSE(l.azRate());
		ASSERT_FALSE(l.elRate());
		ASSERT_FALSE(l.rollRate());

		l = LOS_Option(az, azRate, el, elRate, roll, rollRate);
		ASSERT_EQ(l.az(), az);
		ASSERT_EQ(l.el(), el);
		ASSERT_EQ(l.roll().value(), roll);
		ASSERT_EQ(l.azRate().value(), azRate);
		ASSERT_EQ(l.elRate().value(), elRate);
		ASSERT_EQ(l.rollRate().value(), rollRate);

		l = LOS_Option();
		l.setAz(az);
		l.setEl(el);
		l.setRoll(roll);
		l.setAzRate(azRate);
		l.setElRate(elRate);
		l.setRollRate(rollRate);
		ASSERT_EQ(l.az(), az);
		ASSERT_EQ(l.el(), el);
		ASSERT_EQ(l.roll().value(), roll);
		ASSERT_EQ(l.azRate().value(), azRate);
		ASSERT_EQ(l.elRate().value(), elRate);
		ASSERT_EQ(l.rollRate().value(), rollRate);

		auto l2 = l;
		ASSERT_TRUE(l == l2);
		ASSERT_FALSE(l != l2);
	}

	/**
	 * @brief test LOS
	 */
	TEST_F(TypesTestSuite, TestLOS)
	{
		auto l = LOS();
		ASSERT_EQ(l.bearing(), 0.0);
		ASSERT_EQ(l.el(), 0.0);
		ASSERT_FALSE(l.slantRange());

		double bearing = 1.0;
		double el = 2.0;
		double slantRange = 3.0;

		l = LOS(bearing, el);
		ASSERT_EQ(l.bearing(), bearing);
		ASSERT_EQ(l.el(), el);
		ASSERT_FALSE(l.slantRange());

		l = LOS(bearing, el, slantRange);
		ASSERT_EQ(l.bearing(), bearing);
		ASSERT_EQ(l.el(), el);
		ASSERT_EQ(l.slantRange().value(), slantRange);

		l = LOS();
		l.setBearing(bearing);
		l.setEl(el);
		l.setSlantRange(slantRange);
		ASSERT_EQ(l.bearing(), bearing);
		ASSERT_EQ(l.el(), el);
		ASSERT_EQ(l.slantRange().value(), slantRange);

		auto l2 = l;
		ASSERT_TRUE(l == l2);
		ASSERT_FALSE(l != l2);
	}

	/**
	 * @brief test LOSUncertainty
	 */
	TEST_F(TypesTestSuite, TestLOSUncertainty)
	{
		auto l = LOSUncertainty();
		ASSERT_EQ(l.bearing(), 0.0);
		ASSERT_EQ(l.el(), 0.0);
		ASSERT_FALSE(l.slantRange());

		double bearing = 1.0;
		double el = 2.0;
		double slantRange = 3.0;

		l = LOSUncertainty(bearing, el);
		ASSERT_EQ(l.bearing(), bearing);
		ASSERT_EQ(l.el(), el);
		ASSERT_FALSE(l.slantRange());

		l = LOSUncertainty(bearing, el, slantRange);
		ASSERT_EQ(l.bearing(), bearing);
		ASSERT_EQ(l.el(), el);
		ASSERT_EQ(l.slantRange().value(), slantRange);

		l = LOSUncertainty();
		l.setBearing(bearing);
		l.setEl(el);
		l.setSlantRange(slantRange);
		ASSERT_EQ(l.bearing(), bearing);
		ASSERT_EQ(l.el(), el);
		ASSERT_EQ(l.slantRange().value(), slantRange);

		auto l2 = l;
		ASSERT_TRUE(l == l2);
		ASSERT_FALSE(l != l2);
	}

	/**
	 * @brief test Orientation
	 */
	TEST_F(TypesTestSuite, TestOrientation)
	{
		auto o = Orientation();
		ASSERT_EQ(o.yaw(), 0.0);
		ASSERT_EQ(o.pitch(), 0.0);
		ASSERT_EQ(o.roll(), 0.0);
		ASSERT_FALSE(o.timestamp());

		double yaw = 1.0;
		double pitch = 2.0;
		double roll = 3.0;
		auto timestamp = Clock::now();

		o = Orientation(yaw, pitch, roll);
		ASSERT_EQ(o.yaw(), yaw);
		ASSERT_EQ(o.pitch(), pitch);
		ASSERT_EQ(o.roll(), roll);
		ASSERT_FALSE(o.timestamp());

		o = Orientation(yaw, pitch, roll, timestamp);
		ASSERT_EQ(o.yaw(), yaw);
		ASSERT_EQ(o.pitch(), pitch);
		ASSERT_EQ(o.roll(), roll);
		ASSERT_TRUE(o.timestamp().value() == timestamp);

		o = Orientation();
		o.setRoll(roll);
		o.setPitch(pitch);
		o.setYaw(yaw);
		o.setTimestamp(timestamp);
		ASSERT_EQ(o.yaw(), yaw);
		ASSERT_EQ(o.pitch(), pitch);
		ASSERT_EQ(o.roll(), roll);
		ASSERT_TRUE(o.timestamp().value() == timestamp);

		auto o2 = o;
		ASSERT_TRUE(o == o2);
		ASSERT_FALSE(o != o2);
	}

	/**
	 * @brief test OrientationRate
	 */
	TEST_F(TypesTestSuite, TestOrientationRate)
	{
		auto o = OrientationRate();
		ASSERT_EQ(o.yaw(), 0.0);
		ASSERT_EQ(o.pitch(), 0.0);
		ASSERT_EQ(o.roll(), 0.0);
		ASSERT_FALSE(o.timestamp());

		double yaw = 1.0;
		double pitch = 2.0;
		double roll = 3.0;
		auto timestamp = Clock::now();

		o = OrientationRate(yaw, pitch, roll);
		ASSERT_EQ(o.yaw(), yaw);
		ASSERT_EQ(o.pitch(), pitch);
		ASSERT_EQ(o.roll(), roll);
		ASSERT_FALSE(o.timestamp());

		o = OrientationRate(yaw, pitch, roll, timestamp);
		ASSERT_EQ(o.yaw(), yaw);
		ASSERT_EQ(o.pitch(), pitch);
		ASSERT_EQ(o.roll(), roll);
		ASSERT_TRUE(o.timestamp().value() == timestamp);

		o = OrientationRate();
		o.setRoll(roll);
		o.setPitch(pitch);
		o.setYaw(yaw);
		o.setTimestamp(timestamp);
		ASSERT_EQ(o.yaw(), yaw);
		ASSERT_EQ(o.pitch(), pitch);
		ASSERT_EQ(o.roll(), roll);
		ASSERT_TRUE(o.timestamp().value() == timestamp);

		auto o2 = o;
		ASSERT_TRUE(o == o2);
		ASSERT_FALSE(o != o2);
	}

	/**
	 * @brief test OrientationAcceleration
	 */
	TEST_F(TypesTestSuite, TestOrientationAcceleration)
	{
		auto o = OrientationAcceleration();
		ASSERT_EQ(o.yaw(), 0.0);
		ASSERT_EQ(o.pitch(), 0.0);
		ASSERT_EQ(o.roll(), 0.0);
		ASSERT_FALSE(o.timestamp());

		double yaw = 1.0;
		double pitch = 2.0;
		double roll = 3.0;
		auto timestamp = Clock::now();

		o = OrientationAcceleration(yaw, pitch, roll);
		ASSERT_EQ(o.yaw(), yaw);
		ASSERT_EQ(o.pitch(), pitch);
		ASSERT_EQ(o.roll(), roll);
		ASSERT_FALSE(o.timestamp());

		o = OrientationAcceleration(yaw, pitch, roll, timestamp);
		ASSERT_EQ(o.yaw(), yaw);
		ASSERT_EQ(o.pitch(), pitch);
		ASSERT_EQ(o.roll(), roll);
		ASSERT_TRUE(o.timestamp().value() == timestamp);

		o = OrientationAcceleration();
		o.setRoll(roll);
		o.setPitch(pitch);
		o.setYaw(yaw);
		o.setTimestamp(timestamp);
		ASSERT_EQ(o.yaw(), yaw);
		ASSERT_EQ(o.pitch(), pitch);
		ASSERT_EQ(o.roll(), roll);
		ASSERT_TRUE(o.timestamp().value() == timestamp);

		auto o2 = o;
		ASSERT_TRUE(o == o2);
		ASSERT_FALSE(o != o2);
	}

	/**
	 * @brief test PlatformKinematics
	 */
	TEST_F(TypesTestSuite, TestPlatformKinematics)
	{
		auto p = PlatformKinematics();
		ASSERT_EQ(p.position(), Point4d{});
		ASSERT_EQ(p.velocity(), Velocity3d{});
		ASSERT_FALSE(p.acceleration());
		ASSERT_FALSE(p.orientation());
		ASSERT_FALSE(p.orientationRate());
		ASSERT_FALSE(p.orientationAcceleration());

		auto position = Point4d{1.0, 2.0, 3.0, Clock::now()};
		auto velocity = Velocity3d{4.0, 5.0, 6.0, Clock::now()};
		auto acceleration = Acceleration3d{7.0, 8.0, 9.0, Clock::now()};
		auto orientation = Orientation{10.0, 11.0, 12.0, Clock::now()};
		auto orientationRate = OrientationRate{13.0, 14.0, 15.0, Clock::now()};
		auto orientationAcceleration = OrientationAcceleration{16.0, 17.0, 18.0, Clock::now()};

		p = PlatformKinematics(position);
		ASSERT_EQ(p.position(), position);
		ASSERT_EQ(p.velocity(), Velocity3d{});
		ASSERT_FALSE(p.acceleration());
		ASSERT_FALSE(p.orientation());
		ASSERT_FALSE(p.orientationRate());
		ASSERT_FALSE(p.orientationAcceleration());

		p = PlatformKinematics(position, velocity);
		ASSERT_EQ(p.position(), position);
		ASSERT_EQ(p.velocity(), velocity);
		ASSERT_FALSE(p.acceleration());
		ASSERT_FALSE(p.orientation());
		ASSERT_FALSE(p.orientationRate());
		ASSERT_FALSE(p.orientationAcceleration());

		p = PlatformKinematics(position, velocity, acceleration);
		ASSERT_EQ(p.position(), position);
		ASSERT_EQ(p.velocity(), velocity);
		ASSERT_EQ(p.acceleration().value(), acceleration);
		ASSERT_FALSE(p.orientation());
		ASSERT_FALSE(p.orientationRate());
		ASSERT_FALSE(p.orientationAcceleration());

		p = PlatformKinematics(position, velocity, acceleration, orientation);
		ASSERT_EQ(p.position(), position);
		ASSERT_EQ(p.velocity(), velocity);
		ASSERT_EQ(p.acceleration().value(), acceleration);
		ASSERT_EQ(p.orientation().value(), orientation);
		ASSERT_FALSE(p.orientationRate());
		ASSERT_FALSE(p.orientationAcceleration());

		p = PlatformKinematics(position, velocity, acceleration, orientation, orientationRate);
		ASSERT_EQ(p.position(), position);
		ASSERT_EQ(p.velocity(), velocity);
		ASSERT_EQ(p.acceleration().value(), acceleration);
		ASSERT_EQ(p.orientation().value(), orientation);
		ASSERT_EQ(p.orientationRate().value(), orientationRate);
		ASSERT_FALSE(p.orientationAcceleration());

		p = PlatformKinematics(position, velocity, acceleration, orientation, orientationRate, orientationAcceleration);
		ASSERT_EQ(p.position(), position);
		ASSERT_EQ(p.velocity(), velocity);
		ASSERT_EQ(p.acceleration().value(), acceleration);
		ASSERT_EQ(p.orientation().value(), orientation);
		ASSERT_EQ(p.orientationRate().value(), orientationRate);
		ASSERT_EQ(p.orientationAcceleration().value(), orientationAcceleration);

		p = PlatformKinematics();
		p.setPosition(position);
		p.setVelocity(velocity);
		p.setAcceleration(acceleration);
		p.setOrientation(orientation);
		p.setOrientationRate(orientationRate);
		p.setOrientationAcceleration(orientationAcceleration);
		ASSERT_EQ(p.position(), position);
		ASSERT_EQ(p.velocity(), velocity);
		ASSERT_EQ(p.acceleration().value(), acceleration);
		ASSERT_EQ(p.orientation().value(), orientation);
		ASSERT_EQ(p.orientationRate().value(), orientationRate);
		ASSERT_EQ(p.orientationAcceleration().value(), orientationAcceleration);

		auto p2 = p;
		ASSERT_TRUE(p == p2);
		ASSERT_FALSE(p != p2);
	}

	/**
	 * @brief test Point2d
	 */
	TEST_F(TypesTestSuite, TestPoint2d)
	{
		auto p = Point2d();
		ASSERT_EQ(p.latitude(), 0.0);
		ASSERT_EQ(p.longitude(), 0.0);
		ASSERT_FALSE(p.altitude());
		ASSERT_FALSE(p.timestamp());

		double latitude = 1.0;
		double longitude = 2.0;
		double altitude = 3.0;
		auto timestamp = Clock::now();

		p = Point2d(latitude, longitude);
		ASSERT_EQ(p.latitude(), latitude);
		ASSERT_EQ(p.longitude(), longitude);
		ASSERT_FALSE(p.altitude());
		ASSERT_FALSE(p.timestamp());

		p = Point2d(latitude, longitude, timestamp);
		ASSERT_EQ(p.latitude(), latitude);
		ASSERT_EQ(p.longitude(), longitude);
		ASSERT_FALSE(p.altitude());
		ASSERT_TRUE(p.timestamp().value() == timestamp);

		p = Point2d(latitude, longitude, altitude);
		ASSERT_EQ(p.latitude(), latitude);
		ASSERT_EQ(p.longitude(), longitude);
		ASSERT_EQ(p.altitude().value(), altitude);
		ASSERT_FALSE(p.timestamp());

		p = Point2d(latitude, longitude, altitude, timestamp);
		ASSERT_EQ(p.latitude(), latitude);
		ASSERT_EQ(p.longitude(), longitude);
		ASSERT_EQ(p.altitude().value(), altitude);
		ASSERT_TRUE(p.timestamp().value() == timestamp);

		auto altRange = AltitudeRange(-5.0, 10.0);
		p = Point2d();
		p.setLatitude(latitude);
		p.setLongitude(longitude);
		p.setAltitude(altitude);
		p.setAltitudeRange(altRange);
		p.setTimestamp(timestamp);
		ASSERT_EQ(p.latitude(), latitude);
		ASSERT_EQ(p.longitude(), longitude);
		ASSERT_EQ(p.altitude().value(), altitude);
		ASSERT_TRUE(p.timestamp().value() == timestamp);
		ASSERT_TRUE(p.altitudeRange() == altRange);

		auto p2 = p;
		ASSERT_TRUE(p == p2);
		ASSERT_FALSE(p != p2);
	}

	/**
	 * @brief test InertialAxis
	 */
	TEST_F(TypesTestSuite, TestInertialAxis)
	{
		auto a = InertialAxis();
		ASSERT_EQ(a.position(), 0.0);
		ASSERT_FALSE(a.rate());
		ASSERT_FALSE(a.acceleration());

		double position = 1.0;
		double rate = 2.0;
		double acceleration = 3.0;

		a = InertialAxis(position);
		ASSERT_EQ(a.position(), position);
		ASSERT_FALSE(a.rate());
		ASSERT_FALSE(a.acceleration());

		a = InertialAxis(position, rate);
		ASSERT_EQ(a.position(), position);
		ASSERT_EQ(a.rate().value(), rate);
		ASSERT_FALSE(a.acceleration());

		a = InertialAxis(position, rate, acceleration);
		ASSERT_EQ(a.position(), position);
		ASSERT_EQ(a.rate().value(), rate);
		ASSERT_EQ(a.acceleration().value(), acceleration);

		a = InertialAxis();
		a.setPosition(position);
		a.setRate(rate);
		a.setAcceleration(acceleration);
		ASSERT_EQ(a.position(), position);
		ASSERT_EQ(a.rate().value(), rate);
		ASSERT_EQ(a.acceleration().value(), acceleration);

		auto a2 = a;
		ASSERT_TRUE(a == a2);
		ASSERT_FALSE(a != a2);
	}

	/**
	 * @brief test ObjectState
	 */
	TEST_F(TypesTestSuite, TestObjectState)
	{
		auto r = ObjectStateEnum::NEW;
		std::cout << r;
		SUCCEED();
	}

	/**
	 * @brief test Point2dInertial
	 */
	TEST_F(TypesTestSuite, TestPoint2dInertial)
	{
		auto p = Point2dInertial();
		ASSERT_FALSE(p.timestamp());

		auto az = InertialAxis(1.0, 2.0, 3.0);
		auto el = InertialAxis(4.0, 5.0, 6.0);
		auto timestamp = Clock::now();

		p = Point2dInertial(az, el);
		ASSERT_EQ(p.az(), az);
		ASSERT_EQ(p.el(), el);
		ASSERT_FALSE(p.timestamp());

		p = Point2dInertial(az, el, timestamp);
		ASSERT_EQ(p.az(), az);
		ASSERT_EQ(p.el(), el);
		ASSERT_TRUE(p.timestamp().value() == timestamp);

		p = Point2dInertial();
		p.setAz(az);
		p.setEl(el);
		p.setTimestamp(timestamp);
		ASSERT_EQ(p.az(), az);
		ASSERT_EQ(p.el(), el);
		ASSERT_TRUE(p.timestamp().value() == timestamp);

		auto p2 = p;
		ASSERT_TRUE(p == p2);
		ASSERT_FALSE(p != p2);
	}

	/**
	 * @brief test Point4d
	 */
	TEST_F(TypesTestSuite, TestPoint4d)
	{
		auto p = Point4d();
		ASSERT_EQ(p.latitude(), 0.0);
		ASSERT_EQ(p.longitude(), 0.0);
		ASSERT_EQ(p.altitude(), 0.0);
		ASSERT_EQ(p.timestamp().time_since_epoch().count(), 0);

		double latitude = 1.0;
		double longitude = 2.0;
		double altitude = 3.0;
		auto timestamp = Clock::now();

		p = Point4d(latitude, longitude, altitude, timestamp);
		ASSERT_EQ(p.latitude(), latitude);
		ASSERT_EQ(p.longitude(), longitude);
		ASSERT_EQ(p.altitude(), altitude);
		ASSERT_GT(p.timestamp().time_since_epoch().count(), 0);

		p = Point4d();
		p.setLatitude(latitude);
		p.setLongitude(longitude);
		p.setAltitude(altitude);
		p.setTimestamp(timestamp);
		ASSERT_EQ(p.latitude(), latitude);
		ASSERT_EQ(p.longitude(), longitude);
		ASSERT_EQ(p.altitude(), altitude);
		ASSERT_TRUE(p.timestamp() == timestamp);

		auto p2 = p;
		ASSERT_TRUE(p == p2);
		ASSERT_FALSE(p != p2);
	}

	/**
	 * @brief test PointTarget
	 */
	TEST_F(TypesTestSuite, TestPointTarget)
	{
		auto p = PointTarget();
		ASSERT_FALSE(p.velocity());

		auto point = Point2d{1.0, 2.0};
		auto velocity = Velocity2d{3.0, 4.0};

		p = PointTarget(point);
		ASSERT_EQ(p.point(), point);
		ASSERT_FALSE(p.velocity());

		p = PointTarget(point, velocity);
		ASSERT_EQ(p.point(), point);
		ASSERT_EQ(p.velocity().value(), velocity);

		p = PointTarget();
		p.setPoint(point);
		p.setVelocity(velocity);
		ASSERT_EQ(p.point(), point);
		ASSERT_EQ(p.velocity().value(), velocity);

		auto p2 = p;
		ASSERT_TRUE(p == p2);
		ASSERT_FALSE(p != p2);
	}

	/**
	 * @brief test Rank
	 */
	TEST_F(TypesTestSuite, TestRank)
	{
		auto r = Rank();
		ASSERT_EQ(r.priority(), std::numeric_limits<std::uint16_t>::max());
		ASSERT_EQ(r.precedence(), std::numeric_limits<std::uint16_t>::max());

		const auto priority = std::uint16_t{1};
		const auto precedence = std::uint16_t{2};

		r = Rank(priority);
		ASSERT_EQ(r.priority(), priority);
		ASSERT_EQ(r.precedence(), std::numeric_limits<std::uint16_t>::max());

		r = Rank(priority, precedence);
		ASSERT_EQ(r.priority(), priority);
		ASSERT_EQ(r.precedence(), precedence);

		auto r1 = Rank(priority + 1, precedence);
		ASSERT_FALSE(r == r1);
		ASSERT_TRUE(r != r1);
		ASSERT_TRUE(r < r1);

		auto r2 = Rank(priority, precedence + 1);
		ASSERT_TRUE(r < r2);
	}

	/**
	 * @brief test ReferenceFrame
	 */
	TEST_F(TypesTestSuite, TestReferenceFrame)
	{
		auto r = ReferenceFrame::BODY;
		std::cout << r;
		SUCCEED();
	}

	/**
	 * @brief test RelativeSlantRangeLOS3d
	 */
	TEST_F(TypesTestSuite, TestRelativeSlantRangeLOS3d)
	{
		auto r = RelativeSlantRangeLOS3d();
		ASSERT_EQ(r.range(), 0.0);
		ASSERT_EQ(r.rangeError(), 0.0);
		ASSERT_EQ(r.rangeRate(), 0.0);
		ASSERT_EQ(r.rangeRateError(), 0.0);

		const double range = 1.0;
		const double rangeError = 2.0;
		const double rangeRate = 3.0;
		const double rangeRateError = 4.0;

		r = RelativeSlantRangeLOS3d(range, rangeError);
		ASSERT_EQ(r.range(), range);
		ASSERT_EQ(r.rangeError(), rangeError);
		ASSERT_EQ(r.rangeRate(), 0.0);
		ASSERT_EQ(r.rangeRateError(), 0.0);

		r = RelativeSlantRangeLOS3d(range, rangeError, rangeRate, rangeRateError);
		ASSERT_EQ(r.range(), range);
		ASSERT_EQ(r.rangeError(), rangeError);
		ASSERT_EQ(r.rangeRate(), rangeRate);
		ASSERT_EQ(r.rangeRateError(), rangeRateError);

		r = RelativeSlantRangeLOS3d();
		r.setRange(range);
		r.setRangeError(rangeError);
		r.setRangeRate(rangeRate);
		r.setRangeRateError(rangeRateError);
		ASSERT_EQ(r.range(), range);
		ASSERT_EQ(r.rangeError(), rangeError);
		ASSERT_EQ(r.rangeRate(), rangeRate);
		ASSERT_EQ(r.rangeRateError(), rangeRateError);

		auto r2 = r;
		ASSERT_TRUE(r == r2);
		ASSERT_FALSE(r != r2);
	}

	/**
	 * @brief test Repitition Continuous
	 */
	TEST_F(TypesTestSuite, TestRepititionContinous)
	{
		auto r = repitition::Continuous();
		ASSERT_TRUE(r.maximumInterupt() == Duration{0});

		constexpr auto duration = Duration{100};
		r = repitition::Continuous(duration);
		ASSERT_TRUE(r.maximumInterupt() == duration);

		r = repitition::Continuous();
		r.setMaximumInterupt(duration);
		ASSERT_TRUE(r.maximumInterupt() == duration);
	}

	/**
	 * @brief test Repitition Finite
	 */
	TEST_F(TypesTestSuite, TestRepititionFinite)
	{
		auto r = repitition::Finite();
		ASSERT_EQ(r.numRepititions(), 1);

		constexpr auto numReps = 7;
		r = repitition::Finite(numReps);
		ASSERT_EQ(r.numRepititions(), numReps);

		r = repitition::Finite();
		r.setNumRepititions(numReps);
		ASSERT_EQ(r.numRepititions(), numReps);

		const auto r2 = repitition::Finite(numReps);
		for(int i = 0; i < numReps; ++i)
		{
			ASSERT_EQ(r2.decrementNumRepititions(), numReps - i - 1);
			ASSERT_EQ(r2.numRepititions(), numReps - i - 1);
		}
	}

	/**
	 * @brief test Repitition Periodic
	 */
	TEST_F(TypesTestSuite, TestRepititionPeriodic)
	{
		auto r = repitition::Periodic();
		ASSERT_TRUE(r.interval() == Duration{0});

		constexpr auto interval = Duration{100};
		r = repitition::Periodic(interval);
		ASSERT_TRUE(r.interval() == interval);

		r = repitition::Periodic();
		r.setInterval(interval);
		ASSERT_TRUE(r.interval() == interval);
	}

	/**
	 * @brief test SensorKinematics
	 */
	TEST_F(TypesTestSuite, TestSensorKinematics)
	{
		auto sk = SensorKinematics();
		ASSERT_TRUE(sk.componentID() == UUID{});
		ASSERT_TRUE(sk.orientation() == Orientation{});

		auto compId = GenerateUUID();
		auto orientation = Orientation(0.1, 0.2, 0.3);
		sk = SensorKinematics(compId, orientation);

		ASSERT_TRUE(sk.componentID() == compId);
		ASSERT_TRUE(sk.orientation() == orientation);
	}

	/**
	 * @brief test SignalNavData
	 */
	TEST_F(TypesTestSuite, TestSignalNavData)
	{
		auto snd = SignalNavData();
		ASSERT_TRUE(snd.timestamp() == TimePoint{});
		ASSERT_TRUE(snd.platformKinematics() == PlatformKinematics());

		auto timestamp = Clock::now();
		auto kinematics = PlatformKinematics();
		snd = SignalNavData(timestamp, kinematics);

		ASSERT_TRUE(snd.timestamp() == timestamp);
		ASSERT_TRUE(snd.platformKinematics() == kinematics);
	}

	/**
	 * @brief test Vec3
	 */
	TEST_F(TypesTestSuite, TestVec3)
	{
		double scalar = 5.0;

		auto vec = Vec3{};
		ASSERT_EQ(vec.x(), 0.0);
		ASSERT_EQ(vec.y(), 0.0);
		ASSERT_EQ(vec.z(), 0.0);

		auto vec1 = Vec3(1.0, 2.0, 3.0);
		ASSERT_EQ(vec1.x(), 1.0);
		ASSERT_EQ(vec1.y(), 2.0);
		ASSERT_EQ(vec1.z(), 3.0);

		std::array<double, 3> arr;
		arr[0] = 3.0;
		arr[1] = 5.0;
		arr[2] = 7.0;
		auto vec2 = Vec3(arr);
		ASSERT_EQ(vec2.x(), 3.0);
		ASSERT_EQ(vec2.y(), 5.0);
		ASSERT_EQ(vec2.z(), 7.0);

		auto testVec = vec1.up();
		ASSERT_EQ(testVec.x(), 0.0);
		ASSERT_EQ(testVec.y(), 1.0);
		ASSERT_EQ(testVec.z(), 0.0);

		testVec = vec1.down();
		ASSERT_EQ(testVec.x(), 0.0);
		ASSERT_EQ(testVec.y(), -1.0);
		ASSERT_EQ(testVec.z(), 0.0);

		testVec = vec1.left();
		ASSERT_EQ(testVec.x(), -1.0);
		ASSERT_EQ(testVec.y(), 0.0);
		ASSERT_EQ(testVec.z(), 0.0);

		testVec = vec1.right();
		ASSERT_EQ(testVec.x(), 1.0);
		ASSERT_EQ(testVec.y(), 0.0);
		ASSERT_EQ(testVec.z(), 0.0);

		testVec = vec1.back();
		ASSERT_EQ(testVec.x(), 0.0);
		ASSERT_EQ(testVec.y(), 0.0);
		ASSERT_EQ(testVec.z(), -1.0);

		testVec = vec1.forward();
		ASSERT_EQ(testVec.x(), 0.0);
		ASSERT_EQ(testVec.y(), 0.0);
		ASSERT_EQ(testVec.z(), 1.0);

		double angle = vec1.angle(vec2);
		ASSERT_EQ(angle, 0.071919559413184009);

		testVec = vec2.clampMagnitude(scalar);
		ASSERT_EQ(testVec.x(), 1.6464638998453551);
		ASSERT_EQ(testVec.y(), 2.7441064997422586);
		ASSERT_EQ(testVec.z(), 3.8417490996391619);

		testVec = vec1.cross(vec2);
		ASSERT_EQ(testVec.x(), -1.0);
		ASSERT_EQ(testVec.y(), 2.0);
		ASSERT_EQ(testVec.z(), -1.0);

		double distance = vec1.distance(vec2);
		ASSERT_EQ(distance, 5.3851648071345037);

		double dot = vec1.dot(vec2);
		ASSERT_EQ(dot, 34);

		double magnitude = vec1.norm();
		ASSERT_EQ(magnitude, 3.7416573867739409);

		testVec = vec1.normalize();
		ASSERT_EQ(testVec.x(), 0.2672612419124244);
		ASSERT_EQ(testVec.y(), 0.53452248382484879);
		ASSERT_EQ(testVec.z(), 0.8017837257372733);

		testVec = vec1.project(vec2);
		ASSERT_EQ(testVec.x(), 1.2289156626506021);
		ASSERT_EQ(testVec.y(), 2.0481927710843371);
		ASSERT_EQ(testVec.z(), 2.8674698795180715);

		testVec = vec1 += scalar;
		ASSERT_EQ(testVec.x(), 6.0);
		ASSERT_EQ(testVec.y(), 7.0);
		ASSERT_EQ(testVec.z(), 8.0);

		testVec = vec1 += vec2;
		ASSERT_EQ(testVec.x(), 9.0);
		ASSERT_EQ(testVec.y(), 12.0);
		ASSERT_EQ(testVec.z(), 15.0);

		testVec = vec1 -= scalar;
		ASSERT_EQ(testVec.x(), 4.0);
		ASSERT_EQ(testVec.y(), 7.0);
		ASSERT_EQ(testVec.z(), 10.0);

		testVec = vec1 -= vec2;
		ASSERT_EQ(testVec.x(), 1.0);
		ASSERT_EQ(testVec.y(), 2.0);
		ASSERT_EQ(testVec.z(), 3.0);

		testVec = vec1 *= scalar;
		ASSERT_EQ(testVec.x(), 5.0);
		ASSERT_EQ(testVec.y(), 10.0);
		ASSERT_EQ(testVec.z(), 15.0);

		testVec = vec1 /= scalar;
		ASSERT_EQ(testVec.x(), 1.0);
		ASSERT_EQ(testVec.y(), 2.0);
		ASSERT_EQ(testVec.z(), 3.0);

		testVec = vec1 + scalar;
		ASSERT_EQ(testVec.x(), 6.0);
		ASSERT_EQ(testVec.y(), 7.0);
		ASSERT_EQ(testVec.z(), 8.0);

		testVec = scalar + vec1;
		ASSERT_EQ(testVec.x(), 6.0);
		ASSERT_EQ(testVec.y(), 7.0);
		ASSERT_EQ(testVec.z(), 8.0);

		testVec = vec1 + vec2;
		ASSERT_EQ(testVec.x(), 4.0);
		ASSERT_EQ(testVec.y(), 7.0);
		ASSERT_EQ(testVec.z(), 10.0);

		testVec = vec1 - scalar;
		ASSERT_EQ(testVec.x(), -4.0);
		ASSERT_EQ(testVec.y(), -3.0);
		ASSERT_EQ(testVec.z(), -2.0);

		testVec = scalar - vec1;
		ASSERT_EQ(testVec.x(), 4.0);
		ASSERT_EQ(testVec.y(), 3.0);
		ASSERT_EQ(testVec.z(), 2.0);

		testVec = vec1 - vec2;
		ASSERT_EQ(testVec.x(), -2.0);
		ASSERT_EQ(testVec.y(), -3.0);
		ASSERT_EQ(testVec.z(), -4.0);

		testVec = vec1 * scalar;
		ASSERT_EQ(testVec.x(), 5.0);
		ASSERT_EQ(testVec.y(), 10.0);
		ASSERT_EQ(testVec.z(), 15.0);

		testVec = scalar * vec1;
		ASSERT_EQ(testVec.x(), 5.0);
		ASSERT_EQ(testVec.y(), 10.0);
		ASSERT_EQ(testVec.z(), 15.0);

		double dotProd = vec1 * vec2;
		ASSERT_EQ(dotProd, 34.0);

		testVec = vec1 / scalar;
		ASSERT_EQ(testVec.x(), 0.20000000000000001);
		ASSERT_EQ(testVec.y(), 0.40000000000000002);
		ASSERT_EQ(testVec.z(), 0.59999999999999998);

		testVec = -vec1;
		ASSERT_EQ(testVec.x(), -1.0);
		ASSERT_EQ(testVec.y(), -2.0);
		ASSERT_EQ(testVec.z(), -3.0);

		testVec = vec1;
		ASSERT_TRUE(testVec == vec1);
		ASSERT_FALSE(testVec != vec1);
	}

	/**
	 * @brief test Velocity2d
	 */
	TEST_F(TypesTestSuite, TestVelocity2d)
	{
		auto v = Velocity2d();
		ASSERT_EQ(v.north(), 0.0);
		ASSERT_EQ(v.east(), 0.0);
		ASSERT_FALSE(v.down());
		ASSERT_FALSE(v.timestamp());

		double north = 1.0;
		double east = 2.0;
		double down = 3.0;
		auto timestamp = Clock::now();

		v = Velocity2d(north, east);
		ASSERT_EQ(v.north(), north);
		ASSERT_EQ(v.east(), east);
		ASSERT_FALSE(v.down());
		ASSERT_FALSE(v.timestamp());

		v = Velocity2d(north, east, timestamp);
		ASSERT_EQ(v.north(), north);
		ASSERT_EQ(v.east(), east);
		ASSERT_FALSE(v.down());
		ASSERT_TRUE(v.timestamp().value() == timestamp);

		v = Velocity2d(north, east, down);
		ASSERT_EQ(v.north(), north);
		ASSERT_EQ(v.east(), east);
		ASSERT_EQ(v.down().value(), down);
		ASSERT_FALSE(v.timestamp());

		v = Velocity2d(north, east, down, timestamp);
		ASSERT_EQ(v.north(), north);
		ASSERT_EQ(v.east(), east);
		ASSERT_EQ(v.down().value(), down);
		ASSERT_TRUE(v.timestamp().value() == timestamp);

		v = Velocity2d();
		v.setNorth(north);
		v.setEast(east);
		v.setDown(down);
		v.setTimestamp(timestamp);
		ASSERT_EQ(v.north(), north);
		ASSERT_EQ(v.east(), east);
		ASSERT_EQ(v.down().value(), down);
		ASSERT_TRUE(v.timestamp().value() == timestamp);

		auto v2 = v;
		ASSERT_TRUE(v == v2);
		ASSERT_FALSE(v != v2);
	}

	/**
	 * @brief test Velocity3d
	 */
	TEST_F(TypesTestSuite, TestVelocity3d)
	{
		auto v = Velocity3d();
		ASSERT_EQ(v.north(), 0.0);
		ASSERT_EQ(v.east(), 0.0);
		ASSERT_FALSE(v.down());
		ASSERT_FALSE(v.timestamp());

		double north = 1.0;
		double east = 2.0;
		double down = 3.0;
		auto timestamp = Clock::now();

		v = Velocity3d(north, east, down);
		ASSERT_EQ(v.north(), north);
		ASSERT_EQ(v.east(), east);
		ASSERT_EQ(v.down(), down);
		ASSERT_FALSE(v.timestamp());

		v = Velocity3d(north, east, down, timestamp);
		ASSERT_EQ(v.north(), north);
		ASSERT_EQ(v.east(), east);
		ASSERT_EQ(v.down(), down);
		ASSERT_TRUE(v.timestamp().value() == timestamp);

		v = Velocity3d();
		v.setNorth(north);
		v.setEast(east);
		v.setDown(down);
		v.setTimestamp(timestamp);
		ASSERT_EQ(v.north(), north);
		ASSERT_EQ(v.east(), east);
		ASSERT_EQ(v.down(), down);
		ASSERT_TRUE(v.timestamp().value() == timestamp);

		auto v2 = v;
		ASSERT_TRUE(v == v2);
		ASSERT_FALSE(v != v2);
	}
} // namespace ball::common::utils::math
// UNCLASSIFIED
