cmake_minimum_required(VERSION 3.16)
project(transit_pass LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(
    transit_pass_tests
    transit_pass.cpp
    transit_pass_test.cpp
)

target_compile_features(transit_pass_tests PRIVATE cxx_std_17)
target_compile_options(
    transit_pass_tests
    PRIVATE -Wall -Wextra -Wpedantic -Werror
)
target_link_libraries(transit_pass_tests PRIVATE Threads::Threads)

enable_testing()
add_test(NAME transit_pass COMMAND transit_pass_tests)
