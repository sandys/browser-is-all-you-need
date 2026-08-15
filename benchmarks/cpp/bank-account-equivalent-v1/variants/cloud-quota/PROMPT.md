# Cloud quota

Implement a thread-safe quota bucket that manages an integer
quantity of quota units.

## Lifecycle and operations

A default-constructed quota bucket is inactive. Calling
`provision()` activates it and starts its value at zero. Calling
`provision()` while it is already active throws
`std::runtime_error`.

While active:

- `grant(amount)` adds a positive amount.
- `consume(amount)` removes a positive amount.
- `available()` returns the current amount.
- `consume` throws `std::runtime_error` when the requested
  amount exceeds the current amount.
- Both amount-changing operations throw `std::runtime_error` when
  `amount` is zero or negative.

Calling `retire()` deactivates an active
quota bucket. Calling it before activation or after deactivation
throws `std::runtime_error`. Every `available`,
`grant`, or `consume` operation on an inactive
quota bucket also throws `std::runtime_error`.

Reactivating after deactivation starts a fresh value of zero. No amount
from the previous active lifecycle is retained.

## Concurrency

Many threads call `grant` and `consume` on the
same active object. Each public operation must be internally thread-safe:
validate and update shared state atomically under synchronization so no
transaction is lost and no data race occurs.

## C++ interface contract

Preserve this exact public API:

```cpp
namespace cloud_quota {
class quota_bucket {
public:
    quota_bucket();
    void provision();
    void grant(int amount);
    void consume(int amount);
    void retire();
    int available();
};
}  // namespace cloud_quota
```

The object must remain default-constructible exactly as shown. The test
suite constructs it with
`cloud_quota::quota_bucket value{};`.

## File and build contract

Use only `cloud_quota.h` and `cloud_quota.cpp` for the
implementation. The header must be self-contained and contain the class
declaration and all private state required by the implementation. The
source must define those declared members using the correct namespace and
class qualification; it must not declare a second replacement class.

Only those two implementation files are editable during evaluation. The
test and CMake files are fixed. The implementation must compile as C++17
with `-Wall -Wextra -Wpedantic -Werror -pthread`.
