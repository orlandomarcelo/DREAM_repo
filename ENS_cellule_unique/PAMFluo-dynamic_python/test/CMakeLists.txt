cmake_minimum_required(VERSION 2.6)
project(PAMFluo)

set(pam_VERSION_MAJOR 0)
set(pam_VERSION_MINOR 1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default option is off.
# We override this on the server for a test buid, so tests should be run with this on before checking in.
option(ADDRESS_SANITISER_BUILD "Enable build settings to allow address sanitiser" ON) # default to off

if(ADDRESS_SANITISER_BUILD)

    message(WARNING "Enabling GCC address sanitiser will disable all Valgrind tasks due to compatibility errors")
    set( PROJECT_SANITISE_FLAGS
            "-fsanitize=address,undefined,shift,integer-divide-by-zero,unreachable,vla-bound,null,return,signed-integer-overflow,bounds,alignment,object-size,float-divide-by-zero,float-cast-overflow,nonnull-attribute,returns-nonnull-attribute,bool,enum,vptr")
else()
    MESSAGE("libr sanitize flags off")
    set( PROJECT_SANITISE_FLAGS "")
endif()

# For now everything is debug. Set Compile options locally to maintain independent library builds.
# TBD: add -Wconversion flag.
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic  -Werror -Wno-unused-parameter ${PROJECT_SANITISE_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-O0")
set(CMAKE_C_FLAGS "-Wall -Wextra -Wpedantic  -Werror -Wno-unused-parameter ${PROJECT_SANITISE_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "-O0")

set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${PROJECT_SANITISE_FLAGS}")

############################################################

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

############################################################

set(SOURCES
        tests_main.cpp
        activity_manager_tests.cpp
        ../ActivityManager.cpp
        ../ActivityManager.h
        ../PeriodicActivity.h
        )


add_executable(PAMTest ${SOURCES})
target_link_libraries(PAMTest gtest gmock)

include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/.." )
add_test(
        NAME PAMTest
        COMMAND PAMTest
)