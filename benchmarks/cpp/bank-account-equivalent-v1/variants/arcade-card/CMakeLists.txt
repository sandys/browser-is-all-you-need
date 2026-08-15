cmake_minimum_required(VERSION 3.16)
project(arcade_card LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(
    arcade_card_tests
    arcade_card.cpp
    arcade_card_test.cpp
)

target_compile_features(arcade_card_tests PRIVATE cxx_std_17)
target_compile_options(
    arcade_card_tests
    PRIVATE -Wall -Wextra -Wpedantic -Werror
)
target_link_libraries(arcade_card_tests PRIVATE Threads::Threads)

enable_testing()
add_test(NAME arcade_card COMMAND arcade_card_tests)
