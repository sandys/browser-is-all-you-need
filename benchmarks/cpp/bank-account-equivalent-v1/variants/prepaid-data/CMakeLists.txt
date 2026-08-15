cmake_minimum_required(VERSION 3.16)
project(prepaid_data LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(
    prepaid_data_tests
    prepaid_data.cpp
    prepaid_data_test.cpp
)

target_compile_features(prepaid_data_tests PRIVATE cxx_std_17)
target_compile_options(
    prepaid_data_tests
    PRIVATE -Wall -Wextra -Wpedantic -Werror
)
target_link_libraries(prepaid_data_tests PRIVATE Threads::Threads)

enable_testing()
add_test(NAME prepaid_data COMMAND prepaid_data_tests)
