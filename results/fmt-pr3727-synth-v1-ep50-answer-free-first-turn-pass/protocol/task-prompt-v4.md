# Fix large-date `std::chrono::time_point` formatting

## Reported problem

On Linux, `std::chrono::system_clock::time_point` commonly stores nanoseconds in a signed 64-bit counter and therefore cannot represent dates beyond roughly year 2262. Applications can represent later dates by using a system-clock time point with a coarser duration, for example:

```cpp
using time_point =
    std::chrono::time_point<std::chrono::system_clock,
                            std::chrono::milliseconds>;
```

Such a time point can represent year 3000, but fmt currently overflows while formatting it and emits a date around 1830. Standard-library formatting of the same coarse-resolution value produces the expected future date.

A representative reproduction is:

```cpp
using time_point =
    std::chrono::time_point<std::chrono::system_clock,
                            std::chrono::milliseconds>;
time_point value(std::chrono::seconds(32503680000));
auto result = fmt::format("{:%Y-%m-%d}", value);
```

Required result:

```text
3000-01-01
```

## Required behavior

Solve the general conversion problem rather than special-casing this date or formatted output.

The evaluator checks both of these externally observable contracts:

1. The millisecond-resolution system-clock time point above formats exactly as `3000-01-01`.
2. When `FMT_SAFE_DURATION_CAST` is enabled, an unrepresentable extreme integral duration must fail with the existing checked-conversion behavior instead of wrapping or silently succeeding:

   ```cpp
   using years =
       std::chrono::duration<std::int64_t, std::ratio<31556952>>;
   std::chrono::time_point<std::chrono::system_clock, years> value(
       years((std::numeric_limits<std::int64_t>::max)()));
   ```

   Formatting this value must throw `fmt::format_error` with the message `cannot format duration`.

Ordinary dates, dates before the Unix epoch, fractional seconds, local-time formatting, and existing duration formatting must remain correct. The complete existing test suite must stay green.

## Scope and constraints

- The only editable production file is `include/fmt/chrono.h`, supplied to you as `chrono.h`.
- Do not modify tests, build files, generated files, or any other path.
- Keep the header compatible with the project's C++11 baseline.
- Preserve existing preprocessor guards, namespace boundaries, public overload behavior, negative-subsecond handling, timezone behavior, and failure semantics.
- Reuse compatible facilities and conventions already present in the header rather than introducing a parallel conversion subsystem without need.
- Avoid undefined behavior from signed arithmetic overflow and unsafe intermediate narrowing.
- Do not hardcode the year, timestamp, expected string, platform word size, or a test-only branch.
- Make the smallest coherent production change that satisfies the complete contract.
- Do not change unrelated comments, declarations, formatting, or expressions.

Inspect the supplied header carefully, trace all relevant conversion paths, implement the production fix, and return an applicable edit for `chrono.h`. Do not respond with analysis alone.

####

Use the above instructions to modify the supplied file: chrono.h
Don't change the names of existing functions or classes, as they may be referenced from other code like unit tests, etc.
Only use standard libraries, don't suggest installing any packages.


## Failure-derived integration checks for this fresh attempt

The prior candidate failed for ordinary integration mistakes, not because the behavioral contract changed. Avoid repeating them:

- C++11 is a hard requirement: do not use structured bindings or any C++14/17 syntax.
- A helper declared inside `detail` must be called through the correct namespace from formatters outside `detail`.
- Never add a second overload with the same signature as an existing formatter member. Replace or refactor the existing path coherently.
- Keep every edit inside the supplied `chrono.h`. Do not create nested paths or additional files.
- Make Aider SEARCH blocks exact and unique. Re-read the surrounding source before emitting each replacement.
- Before finishing, audit the resulting header for duplicate declarations, unresolved helper names, unsafe narrowing, and every conversion path named above.

## Aider application check from the latest candidate

The latest response found a plausible conversion direction, but its main replacement was not applied because the SEARCH text dropped the existing leading indentation. The grader therefore saw only an unused helper and no behavior change. For this attempt:

- Copy every SEARCH line byte-for-byte from the supplied header, including all leading spaces.
- Include enough exact surrounding lines to make each SEARCH block unique.
- After proposing edits, ensure every new helper has an applied call site and the old narrowing path was actually replaced.

The latest repair then exposed a source-API constraint visible in the supplied header: `safe_duration_cast` has viable overloads only when source and destination representations are both integral or both floating-point. Do not route mixed floating/integral conversions through that API. Preserve the existing mixed-representation behavior, and remember that an ordinary runtime `if` does not prevent invalid template branches from being instantiated in C++11.

## Conversion-boundary audit

Trace the actual overload selected after each conversion. Merely casting `val` to `time_point<system_clock, seconds>` is insufficient if it is then passed to the existing `gmtime(time_point<system_clock>)`: converting to that parameter's default nanosecond duration can recreate the original overflow before `to_time_t` runs. The calendar conversion must avoid that implicit narrowing boundary while preserving checked failures. Also keep helper result variables in a scope and type that is valid after compile-time dispatch; do not rely on branch-local declarations from a runtime type-trait `if`.

For exact Aider anchoring, the system-clock formatter declaration begins at column zero:

```cpp
template <typename Char, typename Duration>
struct formatter<std::chrono::time_point<std::chrono::system_clock, Duration>,
                 Char> : formatter<std::tm, Char> {
```