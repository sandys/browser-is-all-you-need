cmake_minimum_required(VERSION 3.16)
project(library_credit LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(
    library_credit_tests
    library_credit.cpp
    library_credit_test.cpp
)

target_compile_features(library_credit_tests PRIVATE cxx_std_17)
target_compile_options(
    library_credit_tests
    PRIVATE -Wall -Wextra -Wpedantic -Werror
)
target_link_libraries(library_credit_tests PRIVATE Threads::Threads)

enable_testing()
add_test(NAME library_credit COMMAND library_credit_tests)
