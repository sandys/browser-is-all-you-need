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


## Required implementation architecture

Implement the following design yourself. This is the decision context; no solution patch is supplied.

### 1. Add one namespace-level duration conversion wrapper

Place it immediately after the existing `safe_duration_cast` namespace closes and before the `FMT_NOMACRO` comment. It must accept generic `To` and `From` duration types and expose two C++11 compile-time-selected overloads:

- If `From::rep` and `To::rep` are both floating-point or both non-floating-point, use the existing `safe_duration_cast::safe_duration_cast<To>(from, ec)` when `FMT_SAFE_DURATION_CAST` is enabled; throw `format_error("cannot format duration")` when `ec` is set. When the macro is disabled, use ordinary `std::chrono::duration_cast<To>`.
- If exactly one representation is floating-point, always use ordinary `std::chrono::duration_cast<To>` because the existing checked overload is not viable for mixed categories.

Use C++11 SFINAE/`enable_if` overload selection. Do not use a runtime type-trait `if`. The `safe_duration_cast` namespace is in the current `fmt` inline namespace, not in `detail`.

Use this exact insertion anchor:

```cpp
}  // namespace safe_duration_cast
#endif

// Prevents expansion of a preceding token as a function-style macro.
```

### 2. Generalize the existing system-clock `gmtime` overload

Replace the current overload that accepts only the default `std::chrono::time_point<std::chrono::system_clock>` with an overload templated on `Duration`. Convert `time_point.time_since_epoch()` to `std::chrono::seconds` through the wrapper from step 1, then call the existing `gmtime(std::time_t)` endpoint with the converted duration's `.count()`. This avoids conversion back to the default nanosecond clock duration.

The exact old block is:

```cpp
inline std::tm gmtime(
    std::chrono::time_point<std::chrono::system_clock> time_point) {
  return gmtime(std::chrono::system_clock::to_time_t(time_point));
}
```

Do not invent `detail::to_time_t`. Do not pass a duration object directly to `gmtime`; pass its `.count()` as `std::time_t`.

### 3. Update the system-clock formatter coherently

Keep the existing formatter and negative-subsecond correction. Make only these semantic substitutions:

- In the fractional path, compute the whole-second part with the wrapper from step 1 rather than raw `duration_cast<std::chrono::seconds>`, and convert the remainder back to `Duration` through the same wrapper.
- Pass the original `val` directly to the generalized `gmtime` overload in both the fractional and whole-second paths. Do not use `time_point_cast<seconds>` and do not create a formatter-local `to_seconds`/`to_time_t` helper.

The formatter declaration begins at column zero:

```cpp
template <typename Char, typename Duration>
struct formatter<std::chrono::time_point<std::chrono::system_clock, Duration>,
                 Char> : formatter<std::tm, Char> {
```

### 4. Editing and final checks

- Output Aider SEARCH/REPLACE blocks immediately; do not spend the completion budget on extended analysis.
- Use several small, exact blocks rather than replacing the entire formatter.
- Copy every SEARCH line byte-for-byte from the supplied header, including leading spaces.
- Change only `include/fmt/chrono.h`.
- C++11 only. No structured bindings, duplicate overloads, undeclared helpers, runtime trait dispatch, or branch-local values used outside their scope.
- Preserve every unrelated line exactly, especially `get_units<Period>()`, preprocessor directives, comments, namespace boundaries, local-time code, and other formatters.
- Before finishing, verify: wrapper declared before use; both SFINAE branches compile; `.count()` reaches `gmtime(std::time_t)`; both formatter paths call `gmtime(val)`; negative subseconds remain unchanged; every SEARCH block is exact and applied.

## Non-negotiable corrections from the last response

The last response was not applied because it emitted plain code snippets instead of Aider SEARCH/REPLACE blocks. Every edit must use exactly this transport shape, with `chrono.h` as the filename:

~~~~text
chrono.h
```cpp
<<<<<<< SEARCH
exact existing lines copied from chrono.h
=======
replacement lines
>>>>>>> REPLACE
```
~~~~

Emit several small blocks in that format and nothing else.

Compiler rules for the code inside those replacements:

- `From` and `To` are already duration types. They have `rep` and `period`; they do not have a nested `duration` type.
- Do not compute boolean trait constants and branch with ordinary `if`. Define two separate overloads whose declarations use `FMT_ENABLE_IF(...)` so only the valid body is instantiated.
- Do not create a namespace named `duration_cast`. Give the wrapper an ordinary function name in the current `fmt` namespace.
- Replace the old default-duration `gmtime` overload with the templated overload; do not keep both implementations.
- Inside the formatter, spell `std::chrono::seconds` explicitly unless an alias is declared in that same scope.
- The subsecond variable must remain mutable because the existing negative correction applies `+=`.
- Preserve the exact existing negative-correction statements; do not rewrite them.

## Applied-candidate corrections: follow literally

The last candidate's blocks applied, but its C++ did not compile. Do not repeat any of these deviations:

1. The conversion wrapper belongs in the current `fmt` namespace, not `detail`.
2. Its overload conditions compare whether `From::rep` and `To::rep` have the same floating-point category; they do **not** compare whether `From` and `To` are the identical duration type.
3. The mixed-category overload returns ordinary `std::chrono::duration_cast<To>(from)` and never sets an error merely because categories differ.
4. The same-category overload is the only overload that calls `safe_duration_cast::safe_duration_cast` under `FMT_SAFE_DURATION_CAST`.
5. The generalized `gmtime` must replace the old default-duration overload, use the new wrapper to produce `std::chrono::seconds`, and pass `static_cast<std::time_t>(converted.count())` to the existing `gmtime(std::time_t)`. Do not retain the old overload and do not use raw `duration_cast` there.
6. In the formatter there is no type named `Rep` and no local `seconds` alias. Use `Duration` and fully spelled `std::chrono::seconds`.
7. The fractional expression must preserve its original shape: a mutable `auto subsecs` receives the conversion back to `Duration` of `epoch` minus the conversion of `epoch` to `std::chrono::seconds`; both conversions use the new wrapper.
8. After the unchanged negative-subsecond block, the fractional branch returns `formatter<std::tm, Char>::do_format(gmtime(val), ctx, &subsecs)`.
9. The whole-second branch remains `formatter<std::tm, Char>::format(gmtime(val), ctx)`. Do not change it to `do_format(..., nullptr)` and do not introduce unused `seconds value` variables.

Emit exactly five small SEARCH/REPLACE edits: wrapper insertion, `gmtime` overload replacement, fractional conversion replacement, fractional `gmtime` call replacement, and whole-second call replacement. Copy SEARCH sides exactly.

## Final anchor and template-syntax corrections

The helper insertion has repeatedly failed because the SEARCH anchor omitted the existing `Usage` comment. Copy this complete old block exactly:

```cpp
}  // namespace safe_duration_cast
#endif

// Prevents expansion of a preceding token as a function-style macro.
// Usage: f FMT_NOMACRO()
#define FMT_NOMACRO
```

The file is already inside the `fmt` inline namespace at this location. Do **not** open `namespace fmt` or `namespace detail`; insert the two wrapper overloads directly between `#endif` and the existing comment.

Each C++11 overload declaration must follow the existing source pattern:

```cpp
template <typename To, typename From,
          FMT_ENABLE_IF(/* category condition */)>
To your_wrapper_name(const From& from) {
```

For the same-category condition, compare `std::is_floating_point<typename From::rep>::value == std::is_floating_point<typename To::rep>::value`. For the mixed-category overload use `!=`. Do not use `std::is_same<From, To>`, `std::is_integral`, or a runtime `if`.

The same-category body must retain the `#if FMT_SAFE_DURATION_CAST` / `#else` split because the `safe_duration_cast` namespace is unavailable when the macro is disabled. The mixed-category body is only `std::chrono::duration_cast`.

Use `std::chrono::seconds` in `gmtime`, call the wrapper unqualified, and call `gmtime(val)` in the formatter so `Duration` is deduced. Do not explicitly write `gmtime<Duration>`.