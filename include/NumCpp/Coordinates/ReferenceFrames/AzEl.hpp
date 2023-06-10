#pragma once

#include <iostream>

#include "NumCpp/Utils/essentiallyEqual.hpp"

namespace nc::coordinates::reference_frames
{
    /**
     * @brief Az El coordinates
     */

    class AzEl
    {
    public:
        double az{ 0. }; // radians
        double el{ 0. }; // radians

        /**
         * @brief Default Constructor
         */
        AzEl() = default;

        /**
         * @brief Constructor
         * @param inAz: az value in radians
         * @param inEl: el value in radians
         */
        // NOTLINTNEXTLINE(bugprone-easily-swappable-parameters)
        constexpr AzEl(double inAz, double inEl) noexcept :
            az(inAz),
            el(inEl)
        {
        }

        /**
         * @brief Non-Equality Operator
         *
         * @param other: other object
         * @return bool true if not equal equal
         */
        bool operator==(const AzEl& other) const noexcept
        {
            return utils::essentiallyEqual(az, other.az) && utils::essentiallyEqual(el, other.el);
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
    };

    /**
     * @brief Stream operator
     *
     * @param: os: the output stream
     * @param: point: the AzEl point
     */
    inline std::ostream& operator<<(std::ostream& os, const AzEl& point)
    {
        os << "AzEl(az=" << point.az << ", el=" << point.el << ")\n";
        return os;
    }

} // namespace nc::coordinates::reference_frames
