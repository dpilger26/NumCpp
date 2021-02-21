#pragma once

#include "NumCpp/Utils.hpp"
#include "TypeTraits.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace nc_develop
{
    namespace utils
    {
        template<typename ContainerType>
        void printContainer(const ContainerType& container)
        {
            std::for_each(container.begin(), container.end(),
                [](const auto value) -> void
                {
                    std::cout << value << ' ';
                }
            );

            std::cout << '\n';
        }

        template<typename ContainerType>
        std::string stringifyContainer(const ContainerType& container)
        {
            std::string returnStr = "[";
            std::for_each(container.begin(), container.end(),
                [&returnStr](const auto value) -> void
                {
                    returnStr += std::to_string(value) + ", ";
                }
            );
            returnStr += ']';
            return returnStr;
        }
    }
}
