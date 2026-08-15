# Reward points

Implement a thread-safe reward account that manages an integer
quantity of points.

## Lifecycle and operations

A default-constructed reward account is inactive. Calling
`open()` activates it and starts its value at zero. Calling
`open()` while it is already active throws
`std::runtime_error`.

While active:

- `earn(amount)` adds a positive amount.
- `redeem(amount)` removes a positive amount.
- `points()` returns the current amount.
- `redeem` throws `std::runtime_error` when the requested
  amount exceeds the current amount.
- Both amount-changing operations throw `std::runtime_error` when
  `amount` is zero or negative.

Calling `close()` deactivates an active
reward account. Calling it before activation or after deactivation
throws `std::runtime_error`. Every `points`,
`earn`, or `redeem` operation on an inactive
reward account also throws `std::runtime_error`.

Reactivating after deactivation starts a fresh value of zero. No amount
from the previous active lifecycle is retained.

## Concurrency

Many threads call `earn` and `redeem` on the
same active object. Each public operation must be internally thread-safe:
validate and update shared state atomically under synchronization so no
transaction is lost and no data race occurs.

## C++ interface contract

Preserve this exact public API:

```cpp
namespace reward_points {
class reward_account {
public:
    reward_account();
    void open();
    void earn(int amount);
    void redeem(int amount);
    void close();
    int points();
};
}  // namespace reward_points
```

The object must remain default-constructible exactly as shown. The test
suite constructs it with
`reward_points::reward_account value{};`.

## File and build contract

Use only `reward_points.h` and `reward_points.cpp` for the
implementation. The header must be self-contained and contain the class
declaration and all private state required by the implementation. The
source must define those declared members using the correct namespace and
class qualification; it must not declare a second replacement class.

Only those two implementation files are editable during evaluation. The
test and CMake files are fixed. The implementation must compile as C++17
with `-Wall -Wextra -Wpedantic -Werror -pthread`.
