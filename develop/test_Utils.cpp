// UNCLASSIFIED

#include "ball/common/utils/Math.h"
#include "gtest/gtest.h"
#include "test.h"

#include <cmath>
#include <limits>
#include <random>

#include "boost/math/constants/constants.hpp"


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

        auto nedT = transforms::ENUtoNED(reference_frames::ENU{ 1.0, 2.0, 3.0 });
        ASSERT_NEAR(nedT.north(), 2.0, eps);
        ASSERT_NEAR(nedT.east(), 1.0, eps);
        ASSERT_NEAR(nedT.down(), -3.0, eps);

        auto enuT = transforms::NEDtoENU(reference_frames::NED{ 1.0, 2.0, 3.0 });
        ASSERT_NEAR(enuT.east(), 2.0, eps);
        ASSERT_NEAR(enuT.north(), 1.0, eps);
        ASSERT_NEAR(enuT.up(), -3.0, eps);

        ecef = transforms::LLAtoECEF(reference_frames::LLA{ 0, 0, 0 });
        ASSERT_NEAR(ecef.x(), constants::EARTH_EQUATORIAL_RADIUS, eps);
        ASSERT_NEAR(ecef.y(), 0.0, transformEps);
        ASSERT_NEAR(ecef.z(), 0.0, transformEps);

        ecef = transforms::LLAtoECEF(reference_frames::LLA{ 0, boost::math::constants::half_pi<double>(), 0 });
        ASSERT_NEAR(ecef.x(), 0.0, transformEps);
        ASSERT_NEAR(ecef.y(), constants::EARTH_EQUATORIAL_RADIUS, transformEps);
        ASSERT_NEAR(ecef.z(), 0.0, transformEps);

        ecef = transforms::LLAtoECEF(reference_frames::LLA{ 0, boost::math::constants::pi<double>(), 0 });
        ASSERT_NEAR(ecef.x(), -constants::EARTH_EQUATORIAL_RADIUS, transformEps);
        ASSERT_NEAR(ecef.y(), 0.0, transformEps);
        ASSERT_NEAR(ecef.z(), 0.0, transformEps);

        ecef = transforms::LLAtoECEF(reference_frames::LLA{ 0, -boost::math::constants::half_pi<double>(), 0 });
        ASSERT_NEAR(ecef.x(), 0.0, transformEps);
        ASSERT_NEAR(ecef.y(), -constants::EARTH_EQUATORIAL_RADIUS, transformEps);
        ASSERT_NEAR(ecef.z(), 0.0, transformEps);

        ecef = transforms::LLAtoECEF(reference_frames::LLA{ boost::math::constants::half_pi<double>(), 0, 0 });
        ASSERT_NEAR(ecef.x(), 0.0, transformEps);
        ASSERT_NEAR(ecef.y(), 0.0, transformEps);
        ASSERT_NEAR(ecef.z(), constants::EARTH_POLAR_RADIUS, transformEps);

        ecef = transforms::LLAtoECEF(reference_frames::LLA{ -boost::math::constants::half_pi<double>(), 0, 0 });
        ASSERT_NEAR(ecef.x(), 0.0, transformEps);
        ASSERT_NEAR(ecef.y(), 0.0, transformEps);
        ASSERT_NEAR(ecef.z(), -constants::EARTH_POLAR_RADIUS, transformEps);

        ecef = transforms::LLAtoECEF(reference_frames::LLA{ 0, 0, constants::EARTH_EQUATORIAL_RADIUS });
        ASSERT_NEAR(ecef.x(), 2 * constants::EARTH_EQUATORIAL_RADIUS, transformEps);
        ASSERT_NEAR(ecef.y(), 0.0, transformEps);
        ASSERT_NEAR(ecef.z(), 0.0, transformEps);

        lla = transforms::ECEFtoLLA(reference_frames::ECEF{ constants::EARTH_EQUATORIAL_RADIUS, 0, 0 });
        ASSERT_NEAR(lla.latitude(), 0.0, transformEps);
        ASSERT_NEAR(lla.longitude(), 0.0, transformEps);
        ASSERT_NEAR(lla.altitude(), 0.0, transformEps);

        lla = transforms::ECEFtoLLA(reference_frames::ECEF{ 0, constants::EARTH_EQUATORIAL_RADIUS, 0 });
        ASSERT_NEAR(lla.latitude(), 0.0, transformEps);
        ASSERT_NEAR(lla.longitude(), boost::math::constants::half_pi<double>(), transformEps);
        ASSERT_NEAR(lla.altitude(), 0.0, transformEps);

        lla = transforms::ECEFtoLLA(reference_frames::ECEF{ 0, 0, constants::EARTH_POLAR_RADIUS });
        ASSERT_NEAR(lla.latitude(), boost::math::constants::half_pi<double>(), transformEps);
        ASSERT_NEAR(lla.longitude(), 0.0, transformEps);
        ASSERT_NEAR(lla.altitude(), 0.0, transformEps);

        lla = transforms::ECEFtoLLA(reference_frames::ECEF{ 0, 0, -constants::EARTH_POLAR_RADIUS });
        ASSERT_NEAR(lla.latitude(), -boost::math::constants::half_pi<double>(), transformEps);
        ASSERT_NEAR(lla.longitude(), 0.0, transformEps);
        ASSERT_NEAR(lla.altitude(), 0.0, transformEps);

        lla = transforms::ECEFtoLLA(reference_frames::ECEF{ 2 * constants::EARTH_EQUATORIAL_RADIUS, 0, 0 });
        ASSERT_NEAR(lla.latitude(), 0.0, transformEps);
        ASSERT_NEAR(lla.longitude(), 0.0, transformEps);
        ASSERT_NEAR(lla.altitude(), constants::EARTH_EQUATORIAL_RADIUS, transformEps);

        constexpr double platformAltitude = 100.0;
        const auto       platform         = reference_frames::LLA{ 0.0, 0.0, platformAltitude };

        auto target = transforms::LLAtoECEF(reference_frames::LLA{ 0.0, 0.017453, platformAltitude });
        enu         = transforms::ECEFtoENU(target, platform);
        ned         = transforms::ECEFtoNED(target, platform);
        ASSERT_GT(enu.east(), 0.0);
        ASSERT_NEAR(enu.north(), 0.0, transformEps);
        ASSERT_LT(enu.up(), 0.0);
        ASSERT_NEAR(enu.east(), ned.east(), transformEps);
        ASSERT_NEAR(enu.north(), ned.north(), transformEps);
        ASSERT_NEAR(enu.up(), -ned.down(), transformEps);

        target = transforms::LLAtoECEF(reference_frames::LLA{ 0.017453, 0.0, platformAltitude });
        enu    = transforms::ECEFtoENU(target, platform);
        ned    = transforms::ECEFtoNED(target, platform);
        ASSERT_NEAR(enu.east(), 0.0, transformEps);
        ASSERT_GT(enu.north(), 0.0);
        ASSERT_LT(enu.up(), 0.0);
        ASSERT_NEAR(enu.east(), ned.east(), transformEps);
        ASSERT_NEAR(enu.north(), ned.north(), transformEps);
        ASSERT_NEAR(enu.up(), -ned.down(), transformEps);

        target = transforms::LLAtoECEF(reference_frames::LLA{ 0.017453, 0.017453, platformAltitude });
        enu    = transforms::ECEFtoENU(target, platform);
        ned    = transforms::ECEFtoNED(target, platform);
        ASSERT_GT(enu.east(), 0.0);
        ASSERT_GT(enu.north(), 0.0);
        ASSERT_LT(enu.up(), 0.0);
        ASSERT_NEAR(enu.east(), ned.east(), transformEps);
        ASSERT_NEAR(enu.north(), ned.north(), transformEps);
        ASSERT_NEAR(enu.up(), -ned.down(), transformEps);

        target = transforms::LLAtoECEF(reference_frames::LLA{ 0.017453, 0.017453, platformAltitude + 10000 });
        enu    = transforms::ECEFtoENU(target, platform);
        ned    = transforms::ECEFtoNED(target, platform);
        ASSERT_GT(enu.east(), 0.0);
        ASSERT_GT(enu.north(), 0.0);
        ASSERT_GT(enu.up(), 0.0);
        ASSERT_NEAR(enu.east(), ned.east(), transformEps);
        ASSERT_NEAR(enu.north(), ned.north(), transformEps);
        ASSERT_NEAR(enu.up(), -ned.down(), transformEps);

        auto targetLLA = reference_frames::LLA{ 0.017453, 0.0, platformAltitude };
        azEl           = transforms::LLAtoAzElGeocentric(targetLLA, platform);
        ASSERT_NEAR(azEl.az(), 0.0, transformEps);
        ASSERT_LT(azEl.el(), 0.0);

        targetLLA = reference_frames::LLA{ 0.0, 0.017453, platformAltitude };
        azEl      = transforms::LLAtoAzElGeocentric(targetLLA, platform);
        ASSERT_NEAR(azEl.az(), boost::math::constants::half_pi<double>(), transformEps);
        ASSERT_LT(azEl.el(), 0.0);

        targetLLA = reference_frames::LLA{ -0.017453, 0.0, platformAltitude };
        azEl      = transforms::LLAtoAzElGeocentric(targetLLA, platform);
        ASSERT_NEAR(azEl.az(), boost::math::constants::pi<double>(), transformEps);
        ASSERT_LT(azEl.el(), 0.0);

        targetLLA = reference_frames::LLA{ 0.0, -0.017453, platformAltitude };
        azEl      = transforms::LLAtoAzElGeocentric(targetLLA, platform);
        ASSERT_NEAR(azEl.az(),
                    boost::math::constants::half_pi<double>() + boost::math::constants::pi<double>(),
                    transformEps);
        ASSERT_LT(azEl.el(), 0.0);

        targetLLA = reference_frames::LLA{ 0.017453, 0.0, platformAltitude + 10000 };
        azEl      = transforms::LLAtoAzElGeocentric(targetLLA, platform);
        ASSERT_NEAR(azEl.az(), 0.0, transformEps);
        ASSERT_GT(azEl.el(), 0.0);

        constexpr int NUM_POINTS = 100;

        auto rng = std::mt19937_64{};
        rng.seed(666);
        std::uniform_real_distribution<double> distLat(-boost::math::constants::half_pi<double>(),
                                                       boost::math::constants::half_pi<double>());
        std::uniform_real_distribution<double> distLon(-boost::math::constants::pi<double>(),
                                                       boost::math::constants::pi<double>());
        std::uniform_real_distribution<double> distAlt(100, 10000);

        for (int i = 0; i < NUM_POINTS; ++i)
        {
            const auto platformPosition = reference_frames::LLA{ distLat(rng), distLon(rng), distAlt(rng) };
            const auto targetPosition   = reference_frames::LLA{ distLat(rng), distLon(rng), distAlt(rng) };

            const auto azEl1 = transforms::LLAtoAzElGeocentric(targetPosition, platformPosition);
            const auto azEl2 = transforms::LLAtoAzElGeodetic(targetPosition, platformPosition);
            ASSERT_NEAR(azEl1.az(), azEl2.az(), 0.1);
            ASSERT_NEAR(azEl1.el(), azEl2.el(), 0.1);
        }

        // AFSIM outputs
        const auto platform2ECEF  = reference_frames::ECEF{ 889780.8040509718, -5443884.478448521, 3191301.5726495585 };
        const auto platform2Euler = Euler{ 1.678885817527771, -1.0427558422088623, -3.0950019359588623 };
        const auto platform2RollPitchYaw = Orientation(0.027159271086905079, 0., 0.);

        const auto platform2RollPitchYawCalc = transforms::ECEFEulerToNEDRollPitchYaw(platform2ECEF, platform2Euler);
        ASSERT_NEAR(platform2RollPitchYawCalc.roll(), platform2RollPitchYaw.roll(), 1e-7);
        ASSERT_NEAR(platform2RollPitchYawCalc.pitch(), platform2RollPitchYaw.pitch(), 1e-7);
        ASSERT_NEAR(platform2RollPitchYawCalc.yaw(), platform2RollPitchYaw.yaw(), 1e-7);

        const auto platform2EulerCalc = transforms::NEDRollPitchYawToECEFEuler(platform2ECEF, platform2RollPitchYaw);
        ASSERT_NEAR(platform2EulerCalc.psi(), platform2Euler.psi(), 1e-7);
        ASSERT_NEAR(platform2EulerCalc.theta(), platform2Euler.theta(), 1e-7);
        ASSERT_NEAR(platform2EulerCalc.phi(), platform2Euler.phi(), 1e-7);

        const auto platform3ECEF = reference_frames::ECEF{ -1288345.7521444533, -4718928.642526492, 4079259.935028878 };
        const auto platform3Euler        = Euler{ 1.30427503581543, -.872403085231781, 3.1415927410125732 };
        const auto platform3RollPitchYaw = Orientation(0., 0., 0.);

        const auto platform3RollPitchYawCalc = transforms::ECEFEulerToNEDRollPitchYaw(platform3ECEF, platform3Euler);
        ASSERT_NEAR(platform3RollPitchYawCalc.roll(), platform3RollPitchYaw.roll(), 1e-7);
        ASSERT_NEAR(platform3RollPitchYawCalc.pitch(), platform3RollPitchYaw.pitch(), 1e-7);
        ASSERT_NEAR(platform3RollPitchYawCalc.yaw(), platform3RollPitchYaw.yaw(), 1e-7);

        const auto platform3EulerCalc = transforms::NEDRollPitchYawToECEFEuler(platform3ECEF, platform3RollPitchYaw);
        ASSERT_NEAR(platform3EulerCalc.psi(), platform3Euler.psi(), 1e-7);
        ASSERT_NEAR(platform3EulerCalc.theta(), platform3Euler.theta(), 1e-7);
        ASSERT_NEAR(platform3EulerCalc.phi(), -platform3Euler.phi(), 1e-7);

        const auto platform4ECEF  = reference_frames::ECEF{ 861284.8918511268, -5441200.936501232, 3203589.383938122 };
        const auto platform4Euler = Euler{ -2.4969322681427, -0.4192129075527191, 2.2737600803375244 };
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
        const auto platformECEF        = transforms::LLAtoECEF(reference_frames::LLA{ 0., 0., 0. });
        auto       platformOrientation = types::Orientation{ 0., 0., 0. };

        auto bodyAzEl = transforms::ECEFtoBody(platformECEF,
                                               platformOrientation,
                                               transforms::LLAtoECEF(reference_frames::LLA{ 0.01, 0., 0. }));
        ASSERT_NEAR(bodyAzEl.az().position(), 0., 1e-7);
        ASSERT_NEAR(bodyAzEl.el().position(), 0., 0.1);

        bodyAzEl = transforms::ECEFtoBody(platformECEF,
                                          platformOrientation,
                                          transforms::LLAtoECEF(reference_frames::LLA{ -0.01, 0., 0. }));
        ASSERT_NEAR(bodyAzEl.az().position(), boost::math::constants::pi<double>(), 1e-7);
        ASSERT_NEAR(bodyAzEl.el().position(), 0., 0.1);

        bodyAzEl = transforms::ECEFtoBody(platformECEF,
                                          platformOrientation,
                                          transforms::LLAtoECEF(reference_frames::LLA{ 0., 0.1, 0. }));
        ASSERT_NEAR(bodyAzEl.az().position(), boost::math::constants::half_pi<double>(), 1e-7);
        ASSERT_NEAR(bodyAzEl.el().position(), 0., 0.1);

        bodyAzEl = transforms::ECEFtoBody(platformECEF,
                                          platformOrientation,
                                          transforms::LLAtoECEF(reference_frames::LLA{ 0., -0.1, 0. }));
        ASSERT_NEAR(bodyAzEl.az().position(),
                    boost::math::constants::pi<double>() + boost::math::constants::half_pi<double>(),
                    1e-7);
        ASSERT_NEAR(bodyAzEl.el().position(), 0., 0.1);

        platformOrientation = types::Orientation{ boost::math::constants::half_pi<double>(), 0., 0. };
        bodyAzEl            = transforms::ECEFtoBody(platformECEF,
                                          platformOrientation,
                                          transforms::LLAtoECEF(reference_frames::LLA{ 0.01, 0., 0. }));
        ASSERT_NEAR(bodyAzEl.az().position(),
                    boost::math::constants::pi<double>() + boost::math::constants::half_pi<double>(),
                    1e-7);
        ASSERT_NEAR(bodyAzEl.el().position(), 0., 0.1);

        platformOrientation = types::Orientation{ -boost::math::constants::half_pi<double>(), 0., 0. };
        bodyAzEl            = transforms::ECEFtoBody(platformECEF,
                                          platformOrientation,
                                          transforms::LLAtoECEF(reference_frames::LLA{ 0.01, 0., 0. }));
        ASSERT_NEAR(bodyAzEl.az().position(), boost::math::constants::half_pi<double>(), 1e-7);
        ASSERT_NEAR(bodyAzEl.el().position(), 0., 0.1);

        platformOrientation = types::Orientation{ 0., boost::math::constants::half_pi<double>() / 2., 0. };
        bodyAzEl            = transforms::ECEFtoBody(platformECEF,
                                          platformOrientation,
                                          transforms::LLAtoECEF(reference_frames::LLA{ 0.01, 0., 0. }));
        ASSERT_NEAR(bodyAzEl.az().position(), 0., 1e-7);
        ASSERT_NEAR(bodyAzEl.el().position(), -boost::math::constants::half_pi<double>() / 2., 0.1);

        platformOrientation = types::Orientation{ 0., -boost::math::constants::half_pi<double>() / 2., 0. };
        bodyAzEl            = transforms::ECEFtoBody(platformECEF,
                                          platformOrientation,
                                          transforms::LLAtoECEF(reference_frames::LLA{ 0.01, 0., 0. }));
        ASSERT_NEAR(bodyAzEl.az().position(), 0., 1e-7);
        ASSERT_NEAR(bodyAzEl.el().position(), boost::math::constants::half_pi<double>() / 2., 0.1);

        platformOrientation = types::Orientation{ 0., 0., boost::math::constants::half_pi<double>() / 2. };
        bodyAzEl            = transforms::ECEFtoBody(platformECEF,
                                          platformOrientation,
                                          transforms::LLAtoECEF(reference_frames::LLA{ 0., 0.01, 0. }));
        ASSERT_NEAR(bodyAzEl.az().position(), boost::math::constants::half_pi<double>(), 1e-7);
        ASSERT_NEAR(bodyAzEl.el().position(), boost::math::constants::half_pi<double>() / 2., 0.1);

        platformOrientation = types::Orientation{ 0., 0., -boost::math::constants::half_pi<double>() / 2. };
        bodyAzEl            = transforms::ECEFtoBody(platformECEF,
                                          platformOrientation,
                                          transforms::LLAtoECEF(reference_frames::LLA{ 0., 0.01, 0. }));
        ASSERT_NEAR(bodyAzEl.az().position(), boost::math::constants::half_pi<double>(), 1e-7);
        ASSERT_NEAR(bodyAzEl.el().position(), -boost::math::constants::half_pi<double>() / 2., 0.1);
    }

    /**
     * @brief test CartToInert
     */
    TEST_F(UtilsTestSuite, TestCartToInertial)
    {
        auto cart  = Cartesian(1.0, 0.0, 0.0);
        auto point = transforms::CartToInertial(cart);
        ASSERT_NEAR(point.az().position(), 0.0, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(point.el().position(), 0.0, std::numeric_limits<double>::epsilon());

        cart  = Cartesian(0.0, 1.0, 0.0);
        point = transforms::CartToInertial(cart);
        ASSERT_NEAR(point.az().position(),
                    boost::math::constants::half_pi<double>(),
                    std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(point.el().position(), 0.0, std::numeric_limits<double>::epsilon());

        cart  = Cartesian(-1.0, 0.0, 0.0);
        point = transforms::CartToInertial(cart);
        ASSERT_NEAR(point.az().position(),
                    boost::math::constants::pi<double>(),
                    std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(point.el().position(), 0.0, std::numeric_limits<double>::epsilon());

        cart  = Cartesian(0.0, -1.0, 0.0);
        point = transforms::CartToInertial(cart);
        ASSERT_NEAR(point.az().position(),
                    boost::math::constants::pi<double>() + boost::math::constants::half_pi<double>(),
                    std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(point.el().position(), 0.0, std::numeric_limits<double>::epsilon());

        cart  = Cartesian(0.0, 0.0, 1.0);
        point = transforms::CartToInertial(cart);
        ASSERT_NEAR(point.az().position(), 0.0, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(point.el().position(),
                    -boost::math::constants::half_pi<double>(),
                    std::numeric_limits<double>::epsilon());

        cart  = Cartesian(0.0, 0.0, -1.0);
        point = transforms::CartToInertial(cart);
        ASSERT_NEAR(point.az().position(), 0.0, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(point.el().position(),
                    boost::math::constants::half_pi<double>(),
                    std::numeric_limits<double>::epsilon());
    }

    /**
     * @brief test Inertial to Cartesian coordinates
     */
    TEST_F(UtilsTestSuite, TestInertialToCart)
    {
        auto point = Point2dInertial(InertialAxis(0.0), InertialAxis(0.0));
        auto cart  = transforms::InertialToCart(point);
        ASSERT_NEAR(cart.x(), 1.0, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(cart.y(), 0.0, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(cart.z(), 0.0, std::numeric_limits<double>::epsilon());

        point = Point2dInertial(InertialAxis(boost::math::constants::half_pi<double>()), InertialAxis(0.0));
        cart  = transforms::InertialToCart(point);
        ASSERT_NEAR(cart.x(), 0.0, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(cart.y(), 1.0, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(cart.z(), 0.0, std::numeric_limits<double>::epsilon());

        point = Point2dInertial(InertialAxis(boost::math::constants::pi<double>()), InertialAxis(0.0));
        cart  = transforms::InertialToCart(point);
        ASSERT_NEAR(cart.x(), -1.0, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(cart.y(), 0.0, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(cart.z(), 0.0, std::numeric_limits<double>::epsilon());

        point = Point2dInertial(
            InertialAxis(boost::math::constants::pi<double>() + boost::math::constants::half_pi<double>()),
            InertialAxis(0.0));
        cart = transforms::InertialToCart(point);
        ASSERT_NEAR(cart.x(), 0.0, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(cart.y(), -1.0, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(cart.z(), 0.0, std::numeric_limits<double>::epsilon());

        point = Point2dInertial(InertialAxis(0.0), InertialAxis(-boost::math::constants::half_pi<double>()));
        cart  = transforms::InertialToCart(point);
        ASSERT_NEAR(cart.x(), 0.0, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(cart.y(), 0.0, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(cart.z(), 1.0, std::numeric_limits<double>::epsilon());

        point = Point2dInertial(InertialAxis(0.0), InertialAxis(boost::math::constants::half_pi<double>()));
        cart  = transforms::InertialToCart(point);
        ASSERT_NEAR(cart.x(), 0.0, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(cart.y(), 0.0, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(cart.z(), -1.0, std::numeric_limits<double>::epsilon());

        point = Point2dInertial(InertialAxis(boost::math::constants::half_pi<double>() / 2.),
                                InertialAxis(boost::math::constants::half_pi<double>() / 2.));
        cart  = transforms::InertialToCart(point);
        ASSERT_NEAR(cart.x(), 0.5, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(cart.y(), 0.5, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(cart.z(), -0.7071067811865476, std::numeric_limits<double>::epsilon());
    }
} // namespace ball::common::utils::math
  // UNCLASSIFIED
