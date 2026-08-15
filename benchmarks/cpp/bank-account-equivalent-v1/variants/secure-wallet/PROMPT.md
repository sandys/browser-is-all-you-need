# Secure wallet

Implement a thread-safe wallet that manages an integer
quantity of credits.

## Lifecycle and operations

A default-constructed wallet is inactive. Calling
`activate()` activates it and starts its value at zero. Calling
`activate()` while it is already active throws
`std::runtime_error`.

While active:

- `deposit(amount)` adds a positive amount.
- `withdraw(amount)` removes a positive amount.
- `balance()` returns the current amount.
- `withdraw` throws `std::runtime_error` when the requested
  amount exceeds the current amount.
- Both amount-changing operations throw `std::runtime_error` when
  `amount` is zero or negative.

Calling `deactivate()` deactivates an active
wallet. Calling it before activation or after deactivation
throws `std::runtime_error`. Every `balance`,
`deposit`, or `withdraw` operation on an inactive
wallet also throws `std::runtime_error`.

Reactivating after deactivation starts a fresh value of zero. No amount
from the previous active lifecycle is retained.

## Concurrency

Many threads call `deposit` and `withdraw` on the
same active object. Each public operation must be internally thread-safe:
validate and update shared state atomically under synchronization so no
transaction is lost and no data race occurs.

## C++ interface contract

Preserve this exact public API:

```cpp
namespace secure_wallet {
class wallet_account {
public:
    wallet_account();
    void activate();
    void deposit(int amount);
    void withdraw(int amount);
    void deactivate();
    int balance();
};
}  // namespace secure_wallet
```

The object must remain default-constructible exactly as shown. The test
suite constructs it with
`secure_wallet::wallet_account value{};`.

## File and build contract

Use only `secure_wallet.h` and `secure_wallet.cpp` for the
implementation. The header must be self-contained and contain the class
declaration and all private state required by the implementation. The
source must define those declared members using the correct namespace and
class qualification; it must not declare a second replacement class.

Only those two implementation files are editable during evaluation. The
test and CMake files are fixed. The implementation must compile as C++17
with `-Wall -Wextra -Wpedantic -Werror -pthread`.
