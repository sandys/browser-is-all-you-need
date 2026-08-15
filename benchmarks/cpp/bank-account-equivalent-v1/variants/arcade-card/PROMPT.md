# Arcade card

Implement a thread-safe player card that manages an integer
quantity of arcade credits.

## Lifecycle and operations

A default-constructed player card is inactive. Calling
`issue()` activates it and starts its value at zero. Calling
`issue()` while it is already active throws
`std::runtime_error`.

While active:

- `load(amount)` adds a positive amount.
- `spend(amount)` removes a positive amount.
- `credits()` returns the current amount.
- `spend` throws `std::runtime_error` when the requested
  amount exceeds the current amount.
- Both amount-changing operations throw `std::runtime_error` when
  `amount` is zero or negative.

Calling `revoke()` deactivates an active
player card. Calling it before activation or after deactivation
throws `std::runtime_error`. Every `credits`,
`load`, or `spend` operation on an inactive
player card also throws `std::runtime_error`.

Reactivating after deactivation starts a fresh value of zero. No amount
from the previous active lifecycle is retained.

## Concurrency

Many threads call `load` and `spend` on the
same active object. Each public operation must be internally thread-safe:
validate and update shared state atomically under synchronization so no
transaction is lost and no data race occurs.

## C++ interface contract

Preserve this exact public API:

```cpp
namespace arcade_card {
class player_card {
public:
    player_card();
    void issue();
    void load(int amount);
    void spend(int amount);
    void revoke();
    int credits();
};
}  // namespace arcade_card
```

The object must remain default-constructible exactly as shown. The test
suite constructs it with
`arcade_card::player_card value{};`.

## File and build contract

Use only `arcade_card.h` and `arcade_card.cpp` for the
implementation. The header must be self-contained and contain the class
declaration and all private state required by the implementation. The
source must define those declared members using the correct namespace and
class qualification; it must not declare a second replacement class.

Only those two implementation files are editable during evaluation. The
test and CMake files are fixed. The implementation must compile as C++17
with `-Wall -Wextra -Wpedantic -Werror -pthread`.
