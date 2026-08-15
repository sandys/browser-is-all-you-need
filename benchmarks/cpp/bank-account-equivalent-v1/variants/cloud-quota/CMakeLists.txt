cmake_minimum_required(VERSION 3.16)
project(cloud_quota LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(
    cloud_quota_tests
    cloud_quota.cpp
    cloud_quota_test.cpp
)

target_compile_features(cloud_quota_tests PRIVATE cxx_std_17)
target_compile_options(
    cloud_quota_tests
    PRIVATE -Wall -Wextra -Wpedantic -Werror
)
target_link_libraries(cloud_quota_tests PRIVATE Threads::Threads)

enable_testing()
add_test(NAME cloud_quota COMMAND cloud_quota_tests)
