cmake_minimum_required(VERSION 3.16)
project(workshop_tokens LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(
    workshop_tokens_tests
    workshop_tokens.cpp
    workshop_tokens_test.cpp
)

target_compile_features(workshop_tokens_tests PRIVATE cxx_std_17)
target_compile_options(
    workshop_tokens_tests
    PRIVATE -Wall -Wextra -Wpedantic -Werror
)
target_link_libraries(workshop_tokens_tests PRIVATE Threads::Threads)

enable_testing()
add_test(NAME workshop_tokens COMMAND workshop_tokens_tests)
