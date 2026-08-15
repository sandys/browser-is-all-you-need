# Prepaid data

Implement a thread-safe data wallet that manages an integer
quantity of megabytes.

## Lifecycle and operations

A default-constructed data wallet is inactive. Calling
`connect()` activates it and starts its value at zero. Calling
`connect()` while it is already active throws
`std::runtime_error`.

While active:

- `add_megabytes(amount)` adds a positive amount.
- `use_megabytes(amount)` removes a positive amount.
- `remaining_megabytes()` returns the current amount.
- `use_megabytes` throws `std::runtime_error` when the requested
  amount exceeds the current amount.
- Both amount-changing operations throw `std::runtime_error` when
  `amount` is zero or negative.

Calling `disconnect()` deactivates an active
data wallet. Calling it before activation or after deactivation
throws `std::runtime_error`. Every `remaining_megabytes`,
`add_megabytes`, or `use_megabytes` operation on an inactive
data wallet also throws `std::runtime_error`.

Reactivating after deactivation starts a fresh value of zero. No amount
from the previous active lifecycle is retained.

## Concurrency

Many threads call `add_megabytes` and `use_megabytes` on the
same active object. Each public operation must be internally thread-safe:
validate and update shared state atomically under synchronization so no
transaction is lost and no data race occurs.

## C++ interface contract

Preserve this exact public API:

```cpp
namespace prepaid_data {
class data_wallet {
public:
    data_wallet();
    void connect();
    void add_megabytes(int amount);
    void use_megabytes(int amount);
    void disconnect();
    int remaining_megabytes();
};
}  // namespace prepaid_data
```

The object must remain default-constructible exactly as shown. The test
suite constructs it with
`prepaid_data::data_wallet value{};`.

## File and build contract

Use only `prepaid_data.h` and `prepaid_data.cpp` for the
implementation. The header must be self-contained and contain the class
declaration and all private state required by the implementation. The
source must define those declared members using the correct namespace and
class qualification; it must not declare a second replacement class.

Only those two implementation files are editable during evaluation. The
test and CMake files are fixed. The implementation must compile as C++17
with `-Wall -Wextra -Wpedantic -Werror -pthread`.
