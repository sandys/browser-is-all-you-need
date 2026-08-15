# Inventory ledger

Implement a thread-safe stock ledger that manages an integer
quantity of items.

## Lifecycle and operations

A default-constructed stock ledger is inactive. Calling
`begin()` activates it and starts its value at zero. Calling
`begin()` while it is already active throws
`std::runtime_error`.

While active:

- `receive(amount)` adds a positive amount.
- `dispatch(amount)` removes a positive amount.
- `quantity()` returns the current amount.
- `dispatch` throws `std::runtime_error` when the requested
  amount exceeds the current amount.
- Both amount-changing operations throw `std::runtime_error` when
  `amount` is zero or negative.

Calling `end()` deactivates an active
stock ledger. Calling it before activation or after deactivation
throws `std::runtime_error`. Every `quantity`,
`receive`, or `dispatch` operation on an inactive
stock ledger also throws `std::runtime_error`.

Reactivating after deactivation starts a fresh value of zero. No amount
from the previous active lifecycle is retained.

## Concurrency

Many threads call `receive` and `dispatch` on the
same active object. Each public operation must be internally thread-safe:
validate and update shared state atomically under synchronization so no
transaction is lost and no data race occurs.

## C++ interface contract

Preserve this exact public API:

```cpp
namespace inventory_ledger {
class stock_ledger {
public:
    stock_ledger();
    void begin();
    void receive(int amount);
    void dispatch(int amount);
    void end();
    int quantity();
};
}  // namespace inventory_ledger
```

The object must remain default-constructible exactly as shown. The test
suite constructs it with
`inventory_ledger::stock_ledger value{};`.

## File and build contract

Use only `inventory_ledger.h` and `inventory_ledger.cpp` for the
implementation. The header must be self-contained and contain the class
declaration and all private state required by the implementation. The
source must define those declared members using the correct namespace and
class qualification; it must not declare a second replacement class.

Only those two implementation files are editable during evaluation. The
test and CMake files are fixed. The implementation must compile as C++17
with `-Wall -Wextra -Wpedantic -Werror -pthread`.
