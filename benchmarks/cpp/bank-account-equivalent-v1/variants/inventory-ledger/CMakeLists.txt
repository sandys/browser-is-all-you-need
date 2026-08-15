cmake_minimum_required(VERSION 3.16)
project(inventory_ledger LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(
    inventory_ledger_tests
    inventory_ledger.cpp
    inventory_ledger_test.cpp
)

target_compile_features(inventory_ledger_tests PRIVATE cxx_std_17)
target_compile_options(
    inventory_ledger_tests
    PRIVATE -Wall -Wextra -Wpedantic -Werror
)
target_link_libraries(inventory_ledger_tests PRIVATE Threads::Threads)

enable_testing()
add_test(NAME inventory_ledger COMMAND inventory_ledger_tests)
