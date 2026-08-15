# Transit pass

Implement a thread-safe fare pass that manages an integer
quantity of fare credits.

## Lifecycle and operations

A default-constructed fare pass is inactive. Calling
`activate()` activates it and starts its value at zero. Calling
`activate()` while it is already active throws
`std::runtime_error`.

While active:

- `top_up(amount)` adds a positive amount.
- `charge(amount)` removes a positive amount.
- `funds()` returns the current amount.
- `charge` throws `std::runtime_error` when the requested
  amount exceeds the current amount.
- Both amount-changing operations throw `std::runtime_error` when
  `amount` is zero or negative.

Calling `suspend()` deactivates an active
fare pass. Calling it before activation or after deactivation
throws `std::runtime_error`. Every `funds`,
`top_up`, or `charge` operation on an inactive
fare pass also throws `std::runtime_error`.

Reactivating after deactivation starts a fresh value of zero. No amount
from the previous active lifecycle is retained.

## Concurrency

Many threads call `top_up` and `charge` on the
same active object. Each public operation must be internally thread-safe:
validate and update shared state atomically under synchronization so no
transaction is lost and no data race occurs.

## C++ interface contract

Preserve this exact public API:

```cpp
namespace transit_pass {
class fare_pass {
public:
    fare_pass();
    void activate();
    void top_up(int amount);
    void charge(int amount);
    void suspend();
    int funds();
};
}  // namespace transit_pass
```

The object must remain default-constructible exactly as shown. The test
suite constructs it with
`transit_pass::fare_pass value{};`.

## File and build contract

Use only `transit_pass.h` and `transit_pass.cpp` for the
implementation. The header must be self-contained and contain the class
declaration and all private state required by the implementation. The
source must define those declared members using the correct namespace and
class qualification; it must not declare a second replacement class.

Only those two implementation files are editable during evaluation. The
test and CMake files are fixed. The implementation must compile as C++17
with `-Wall -Wextra -Wpedantic -Werror -pthread`.
