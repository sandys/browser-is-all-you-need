#include "energy_reserve.h"

#include <chrono>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

class check_suite {
public:
    void expect(bool condition, std::string_view description) {
        ++count_;
        if (!condition) {
            failed_ = true;
            std::cerr << "FAILED: " << description << '\n';
        }
    }

    template <typename Function>
    void expect_runtime_error(Function&& function, std::string_view description) {
        ++count_;
        try {
            std::forward<Function>(function)();
        } catch (std::runtime_error const&) {
            return;
        } catch (std::exception const& error) {
            failed_ = true;
            std::cerr << "FAILED: " << description
                      << " threw a different exception: " << error.what() << '\n';
            return;
        }
        failed_ = true;
        std::cerr << "FAILED: " << description
                  << " did not throw std::runtime_error\n";
    }

    std::size_t count() const { return count_; }
    bool failed() const { return failed_; }

private:
    std::size_t count_{0};
    bool failed_{false};
};

}  // namespace

int main() {
    check_suite checks;

    {
        energy_reserve::reserve_meter subject{};
        subject.enable();
        checks.expect(subject.remaining_units() == 0, "newly_started_zero");
    }
    {
        energy_reserve::reserve_meter subject{};
        subject.enable();
        subject.add_units(100);
        checks.expect(subject.remaining_units() == 100, "single_credit");
    }
    {
        energy_reserve::reserve_meter subject{};
        subject.enable();
        subject.add_units(100);
        subject.add_units(50);
        checks.expect(subject.remaining_units() == 150, "multiple_credits");
    }
    {
        energy_reserve::reserve_meter subject{};
        subject.enable();
        subject.add_units(100);
        subject.consume_units(75);
        checks.expect(subject.remaining_units() == 25, "single_debit");
    }
    {
        energy_reserve::reserve_meter subject{};
        subject.enable();
        subject.add_units(100);
        subject.consume_units(80);
        subject.consume_units(20);
        checks.expect(subject.remaining_units() == 0, "multiple_debits");
    }
    {
        energy_reserve::reserve_meter subject{};
        subject.enable();
        subject.add_units(100);
        subject.add_units(110);
        subject.consume_units(200);
        subject.add_units(60);
        subject.consume_units(50);
        checks.expect(subject.remaining_units() == 20, "sequential_operations");
    }
    {
        energy_reserve::reserve_meter subject{};
        subject.enable();
        subject.disable();
        checks.expect_runtime_error(
            [&]() { (void)subject.remaining_units(); },
            "value_after_stop_throws");
    }
    {
        energy_reserve::reserve_meter subject{};
        subject.enable();
        subject.disable();
        checks.expect_runtime_error(
            [&]() { subject.add_units(50); },
            "credit_after_stop_throws");
    }
    {
        energy_reserve::reserve_meter subject{};
        checks.expect_runtime_error(
            [&]() { subject.add_units(50); },
            "credit_before_start_throws");
    }
    {
        energy_reserve::reserve_meter subject{};
        subject.enable();
        subject.disable();
        checks.expect_runtime_error(
            [&]() { subject.consume_units(50); },
            "debit_after_stop_throws");
    }
    {
        energy_reserve::reserve_meter subject{};
        checks.expect_runtime_error(
            [&]() { subject.disable(); },
            "stop_before_start_throws");
    }
    {
        energy_reserve::reserve_meter subject{};
        subject.enable();
        checks.expect_runtime_error(
            [&]() { subject.enable(); },
            "start_twice_throws");
    }
    {
        energy_reserve::reserve_meter subject{};
        subject.enable();
        subject.add_units(50);
        subject.disable();
        subject.enable();
        checks.expect(subject.remaining_units() == 0, "restart_resets_zero");
    }
    {
        energy_reserve::reserve_meter subject{};
        subject.enable();
        subject.add_units(25);
        checks.expect_runtime_error(
            [&]() { subject.consume_units(50); },
            "overdraft_throws");
    }
    {
        energy_reserve::reserve_meter subject{};
        subject.enable();
        subject.add_units(100);
        checks.expect_runtime_error(
            [&]() { subject.consume_units(-50); },
            "negative_debit_throws");
    }
    {
        energy_reserve::reserve_meter subject{};
        subject.enable();
        checks.expect_runtime_error(
            [&]() { subject.add_units(-50); },
            "negative_credit_throws");
    }
    {
        energy_reserve::reserve_meter subject{};
        subject.enable();
        std::vector<std::thread> threads;
        threads.reserve(1000);
        for (int index = 0; index < 1000; ++index) {
            threads.emplace_back([&]() {
                using namespace std::chrono_literals;
                subject.add_units(1);
                std::this_thread::sleep_for(5ms);
                subject.consume_units(1);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        checks.expect(subject.remaining_units() == 0, "concurrent_transactions");
    }
    {
        energy_reserve::reserve_meter subject{};
        subject.enable();
        checks.expect_runtime_error(
            [&]() { subject.consume_units(0); },
            "zero_debit_throws");
    }
    {
        energy_reserve::reserve_meter subject{};
        subject.enable();
        checks.expect_runtime_error(
            [&]() { subject.add_units(0); },
            "zero_credit_throws");
    }

    if (checks.failed() || checks.count() != 19U) {
        std::cerr << "Verification failed after " << checks.count()
                  << " assertions\n";
        return 1;
    }

    std::cout << "All tests passed (19 assertions in 19 test cases)\n";
    return 0;
}
