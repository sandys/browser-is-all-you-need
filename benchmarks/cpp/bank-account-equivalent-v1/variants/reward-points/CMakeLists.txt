cmake_minimum_required(VERSION 3.16)
project(reward_points LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(
    reward_points_tests
    reward_points.cpp
    reward_points_test.cpp
)

target_compile_features(reward_points_tests PRIVATE cxx_std_17)
target_compile_options(
    reward_points_tests
    PRIVATE -Wall -Wextra -Wpedantic -Werror
)
target_link_libraries(reward_points_tests PRIVATE Threads::Threads)

enable_testing()
add_test(NAME reward_points COMMAND reward_points_tests)
