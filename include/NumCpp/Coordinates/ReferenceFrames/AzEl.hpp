#pragma once

#include <iostream>

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
         * @param az: az value in radians
         * @param el: el value in radians
         */
        // NOTLINTNEXTLINE(bugprone-easily-swappable-parameters)
        constexpr AzEl(double az, double el) noexcept :
            az(az),
            el(el)
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
            return az == other.az && el == other.el;
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
