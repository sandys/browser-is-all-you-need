cmake_minimum_required(VERSION 3.16)
project(secure_wallet LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(
    secure_wallet_tests
    secure_wallet.cpp
    secure_wallet_test.cpp
)

target_compile_features(secure_wallet_tests PRIVATE cxx_std_17)
target_compile_options(
    secure_wallet_tests
    PRIVATE -Wall -Wextra -Wpedantic -Werror
)
target_link_libraries(secure_wallet_tests PRIVATE Threads::Threads)

enable_testing()
add_test(NAME secure_wallet COMMAND secure_wallet_tests)
