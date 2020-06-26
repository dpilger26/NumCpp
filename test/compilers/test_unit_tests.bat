@ECHO OFF

SET COMPILER="Visual Studio 16 2019"
SET "TOOLSETS=v141 v142"
SET "BOOST_DIR=C:\libs\boost\"
SET "BOOST_VERSIONS=boost_1_68_0 boost_1_70_0 boost_1_73_0"
SET "ARCHITECTURE=x64"
SET "CONFIG=Release"
SET "WARNING_LEVEL=4"
SET "BUILD_TYPE=Build"
SET "CXX_STANDARDS=14 17 20"
SET "PROJECT_NAME=NumCpp"
SET "NUMCPP_PY_PROJ=%PROJECT_NAME%.vcxproj"
SET "BUILD_DIR=buildVS"

ECHO "Building for Visual Studio"

FOR %%T IN (%TOOLSETS%) DO (
    ECHO "Building for toolset %%T"

    FOR %%B IN (%BOOST_VERSIONS%) DO (
        ECHO "Building for boost version %%B"

        FOR %%S IN (%CXX_STANDARDS%) DO (
            ECHO "Building for C++%%S"

            cd ../src
            del ..\lib\%PROJECT_NAME%.pyd
            rmdir /S /Q %BUILD_DIR%
            mkdir %BUILD_DIR%
            cd %BUILD_DIR%

            "SET BOOST_ROOT=%BOOST_DIR%%%B"

            @ECHO ON
            cmake -G %COMPILER% -A %ARCHITECTURE% -T %%T -DCMAKE_BUILD_TYPE=%CONFIG% -DCMAKE_CXX_STANDARD=%%S -DBOOST_ROOT=%BOOST_ROOT% ..

            @ECHO OFF
            SET "BUILD_INPUTS=/p:Configuration=%CONFIG% /p:WarningLevel=%WARNING_LEVEL% -t:%BUILD_TYPE%"

            @ECHO ON
            msbuild %NUMCPP_PY_PROJ% %BUILD_INPUTS%

            @ECHO OFF
            cd ..\..\pytest
            pytest 
        )
    )
)

cd ..\compilers
