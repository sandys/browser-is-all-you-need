# Library credit

Implement a thread-safe patron account that manages an integer
quantity of borrowing credits.

## Lifecycle and operations

A default-constructed patron account is inactive. Calling
`enroll()` activates it and starts its value at zero. Calling
`enroll()` while it is already active throws
`std::runtime_error`.

While active:

- `add_credit(amount)` adds a positive amount.
- `use_credit(amount)` removes a positive amount.
- `credit()` returns the current amount.
- `use_credit` throws `std::runtime_error` when the requested
  amount exceeds the current amount.
- Both amount-changing operations throw `std::runtime_error` when
  `amount` is zero or negative.

Calling `close()` deactivates an active
patron account. Calling it before activation or after deactivation
throws `std::runtime_error`. Every `credit`,
`add_credit`, or `use_credit` operation on an inactive
patron account also throws `std::runtime_error`.

Reactivating after deactivation starts a fresh value of zero. No amount
from the previous active lifecycle is retained.

## Concurrency

Many threads call `add_credit` and `use_credit` on the
same active object. Each public operation must be internally thread-safe:
validate and update shared state atomically under synchronization so no
transaction is lost and no data race occurs.

## C++ interface contract

Preserve this exact public API:

```cpp
namespace library_credit {
class patron_account {
public:
    patron_account();
    void enroll();
    void add_credit(int amount);
    void use_credit(int amount);
    void close();
    int credit();
};
}  // namespace library_credit
```

The object must remain default-constructible exactly as shown. The test
suite constructs it with
`library_credit::patron_account value{};`.

## File and build contract

Use only `library_credit.h` and `library_credit.cpp` for the
implementation. The header must be self-contained and contain the class
declaration and all private state required by the implementation. The
source must define those declared members using the correct namespace and
class qualification; it must not declare a second replacement class.

Only those two implementation files are editable during evaluation. The
test and CMake files are fixed. The implementation must compile as C++17
with `-Wall -Wextra -Wpedantic -Werror -pthread`.
