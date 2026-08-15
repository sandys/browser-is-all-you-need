#include "inventory_ledger.h"

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
        inventory_ledger::stock_ledger subject{};
        subject.begin();
        checks.expect(subject.quantity() == 0, "newly_started_zero");
    }
    {
        inventory_ledger::stock_ledger subject{};
        subject.begin();
        subject.receive(100);
        checks.expect(subject.quantity() == 100, "single_credit");
    }
    {
        inventory_ledger::stock_ledger subject{};
        subject.begin();
        subject.receive(100);
        subject.receive(50);
        checks.expect(subject.quantity() == 150, "multiple_credits");
    }
    {
        inventory_ledger::stock_ledger subject{};
        subject.begin();
        subject.receive(100);
        subject.dispatch(75);
        checks.expect(subject.quantity() == 25, "single_debit");
    }
    {
        inventory_ledger::stock_ledger subject{};
        subject.begin();
        subject.receive(100);
        subject.dispatch(80);
        subject.dispatch(20);
        checks.expect(subject.quantity() == 0, "multiple_debits");
    }
    {
        inventory_ledger::stock_ledger subject{};
        subject.begin();
        subject.receive(100);
        subject.receive(110);
        subject.dispatch(200);
        subject.receive(60);
        subject.dispatch(50);
        checks.expect(subject.quantity() == 20, "sequential_operations");
    }
    {
        inventory_ledger::stock_ledger subject{};
        subject.begin();
        subject.end();
        checks.expect_runtime_error(
            [&]() { (void)subject.quantity(); },
            "value_after_stop_throws");
    }
    {
        inventory_ledger::stock_ledger subject{};
        subject.begin();
        subject.end();
        checks.expect_runtime_error(
            [&]() { subject.receive(50); },
            "credit_after_stop_throws");
    }
    {
        inventory_ledger::stock_ledger subject{};
        checks.expect_runtime_error(
            [&]() { subject.receive(50); },
            "credit_before_start_throws");
    }
    {
        inventory_ledger::stock_ledger subject{};
        subject.begin();
        subject.end();
        checks.expect_runtime_error(
            [&]() { subject.dispatch(50); },
            "debit_after_stop_throws");
    }
    {
        inventory_ledger::stock_ledger subject{};
        checks.expect_runtime_error(
            [&]() { subject.end(); },
            "stop_before_start_throws");
    }
    {
        inventory_ledger::stock_ledger subject{};
        subject.begin();
        checks.expect_runtime_error(
            [&]() { subject.begin(); },
            "start_twice_throws");
    }
    {
        inventory_ledger::stock_ledger subject{};
        subject.begin();
        subject.receive(50);
        subject.end();
        subject.begin();
        checks.expect(subject.quantity() == 0, "restart_resets_zero");
    }
    {
        inventory_ledger::stock_ledger subject{};
        subject.begin();
        subject.receive(25);
        checks.expect_runtime_error(
            [&]() { subject.dispatch(50); },
            "overdraft_throws");
    }
    {
        inventory_ledger::stock_ledger subject{};
        subject.begin();
        subject.receive(100);
        checks.expect_runtime_error(
            [&]() { subject.dispatch(-50); },
            "negative_debit_throws");
    }
    {
        inventory_ledger::stock_ledger subject{};
        subject.begin();
        checks.expect_runtime_error(
            [&]() { subject.receive(-50); },
            "negative_credit_throws");
    }
    {
        inventory_ledger::stock_ledger subject{};
        subject.begin();
        std::vector<std::thread> threads;
        threads.reserve(1000);
        for (int index = 0; index < 1000; ++index) {
            threads.emplace_back([&]() {
                using namespace std::chrono_literals;
                subject.receive(1);
                std::this_thread::sleep_for(5ms);
                subject.dispatch(1);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        checks.expect(subject.quantity() == 0, "concurrent_transactions");
    }
    {
        inventory_ledger::stock_ledger subject{};
        subject.begin();
        checks.expect_runtime_error(
            [&]() { subject.dispatch(0); },
            "zero_debit_throws");
    }
    {
        inventory_ledger::stock_ledger subject{};
        subject.begin();
        checks.expect_runtime_error(
            [&]() { subject.receive(0); },
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
