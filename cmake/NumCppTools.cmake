function(numcpp_find_version_file search_path version_file_path)
    #find the file with the version
    find_file(VERSION_FILE
              NAMES Version.hpp
              PATHS
              "${CMAKE_CURRENT_SOURCE_DIR}/include/NumCpp/Core/Internal"
              ${search_path}
              NO_DEFAULT_PATH
              )
    #check to see if it got defined
    if (DEFINED VERSION_FILE)
        set(${version_file_path} "${VERSION_FILE}" PARENT_SCOPE)
    endif()
endfunction()

function(numcpp_read_version version_file_path version_val full_version_str)
    set(VERSION_LIST)
    set(FULL_VERSION_STR)

    if (EXISTS "${version_file_path}")
        file(READ ${version_file_path} FILE_STRING)

        string(REGEX MATCH "constexpr[ ]char[ ]VERSION\\[\\][ ]=[ ]\"(.+)\"" _ ${FILE_STRING})
        if (CMAKE_MATCH_COUNT EQUAL 1)
            set(VERSION_STRING ${CMAKE_MATCH_1})
        endif()

        #figure out what type it is
        if (DEFINED VERSION_STRING)
            if (VERSION_STRING MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
                string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$" _ ${VERSION_STRING})
                SET(VERSION_MAJOR ${CMAKE_MATCH_1})
                SET(VERSION_MINOR ${CMAKE_MATCH_2})
                SET(VERSION_PATCH ${CMAKE_MATCH_3})
            else()
                SET(VERSION_MAJOR "0")
                SET(VERSION_MINOR "0")
                SET(VERSION_PATCH "0")
            endif()

            SET(VERSION_LIST "${VERSION_MAJOR}" "${VERSION_MINOR}" "${VERSION_PATCH}")
            SET(FULL_VERSION_STR "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
        endif()
    endif()

    set(${version_val} ${VERSION_LIST} PARENT_SCOPE)
    set(${full_version_str} ${FULL_VERSION_STR} PARENT_SCOPE)
endfunction()
