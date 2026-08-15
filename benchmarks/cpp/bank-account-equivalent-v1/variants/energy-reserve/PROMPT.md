# Energy reserve

Implement a thread-safe reserve meter that manages an integer
quantity of energy units.

## Lifecycle and operations

A default-constructed reserve meter is inactive. Calling
`enable()` activates it and starts its value at zero. Calling
`enable()` while it is already active throws
`std::runtime_error`.

While active:

- `add_units(amount)` adds a positive amount.
- `consume_units(amount)` removes a positive amount.
- `remaining_units()` returns the current amount.
- `consume_units` throws `std::runtime_error` when the requested
  amount exceeds the current amount.
- Both amount-changing operations throw `std::runtime_error` when
  `amount` is zero or negative.

Calling `disable()` deactivates an active
reserve meter. Calling it before activation or after deactivation
throws `std::runtime_error`. Every `remaining_units`,
`add_units`, or `consume_units` operation on an inactive
reserve meter also throws `std::runtime_error`.

Reactivating after deactivation starts a fresh value of zero. No amount
from the previous active lifecycle is retained.

## Concurrency

Many threads call `add_units` and `consume_units` on the
same active object. Each public operation must be internally thread-safe:
validate and update shared state atomically under synchronization so no
transaction is lost and no data race occurs.

## C++ interface contract

Preserve this exact public API:

```cpp
namespace energy_reserve {
class reserve_meter {
public:
    reserve_meter();
    void enable();
    void add_units(int amount);
    void consume_units(int amount);
    void disable();
    int remaining_units();
};
}  // namespace energy_reserve
```

The object must remain default-constructible exactly as shown. The test
suite constructs it with
`energy_reserve::reserve_meter value{};`.

## File and build contract

Use only `energy_reserve.h` and `energy_reserve.cpp` for the
implementation. The header must be self-contained and contain the class
declaration and all private state required by the implementation. The
source must define those declared members using the correct namespace and
class qualification; it must not declare a second replacement class.

Only those two implementation files are editable during evaluation. The
test and CMake files are fixed. The implementation must compile as C++17
with `-Wall -Wextra -Wpedantic -Werror -pthread`.
