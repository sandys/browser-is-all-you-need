cmake_minimum_required(VERSION 3.16)
project(energy_reserve LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(
    energy_reserve_tests
    energy_reserve.cpp
    energy_reserve_test.cpp
)

target_compile_features(energy_reserve_tests PRIVATE cxx_std_17)
target_compile_options(
    energy_reserve_tests
    PRIVATE -Wall -Wextra -Wpedantic -Werror
)
target_link_libraries(energy_reserve_tests PRIVATE Threads::Threads)

enable_testing()
add_test(NAME energy_reserve COMMAND energy_reserve_tests)
